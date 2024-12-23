# your_app_name/consumers.py
import cv2
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio
from ultralytics import YOLO
from app.sort.sort import Sort
from PIL import Image
import numpy as np
import time
from django.core.files.base import ContentFile
from asgiref.sync import sync_to_async
import threading
import json
previous_tracked_ids = set() 

def check_red_box(dict, arr):
    # Normalize the array to lowercase strings
    normalized_arr = [str(item).lower() for item in arr]
    
    for a in dict.keys():
        if str(a).lower() in normalized_arr:
            print("in")
            continue
        else:
            print("last test", a, arr)
            return False

    return True



camera = None
start_time=None
elapsed_time=None
model = YOLO(r"C:\Users\kirub\OneDrive\Desktop\yolo9e.pt")
person_tracker = Sort()
person_frame_count = {}
person_equipment_data = {}
results_data = {'total_people': 0, 'people_with_equipment': 0, 'people_without_equipment': 0, 'safety_index': 0}
  
###############################################################  SAVE IMAGES(VIOLATORS) #############################################################################

def save_violator_instance(person_id, violator):
    from .models import People
    violator_instance = People()
    violator_instance.image.save(f'violator_{person_id}.jpg',violator , save=True)
    violator_instance.save()
def are_bottom_points_within(small_box, big_box, threshold=0.85):
    x1_small, y1_small, x2_small, y2_small = small_box
    x1_big, y1_big, x2_big, y2_big = big_box
    x1_inter = max(x1_small, x1_big)
    y1_inter = max(y1_small, y1_big)
    x2_inter = min(x2_small, x2_big)
    y2_inter = min(y2_small, y2_big)
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height
    small_area = (x2_small - x1_small) * (y2_small - y1_small)

    if small_area == 0:
        return False

    overlap_percentage = intersection_area / small_area
    return overlap_percentage >= threshold
############################################################ ANALYTICS #########################################################################
# Global variables
def get_analytics(results):
    people=[]
    detected_boxes = results[0].boxes.xyxy.cpu().numpy()
    detected_classes = results[0].names
    detected_scores = results[0].boxes.conf.cpu().numpy()
    detected_class_ids = results[0].boxes.cls.cpu().numpy()
    for i, box in enumerate(detected_boxes):
        class_id = int(detected_class_ids[i])
        if detected_classes[class_id] == 'person' and detected_scores[i] > 0.45:
            x1, y1, x2, y2 = box
            score = detected_scores[i]
            people.append([x1, y1, x2, y2, score])

    if len(people) > 0:
        people = np.array(people)
    else:
        people = np.empty((0, 5))
    with open('app/compliances.json', 'r') as file:
        data = json.load(file)
    equipment_boxes = {compliance.lower(): [] for compliance in data['compliances']}



    for i, box in enumerate(detected_boxes):
        class_id = int(detected_class_ids[i])
        score = detected_scores[i]
        
        if detected_classes[class_id] in equipment_boxes and score > 0.20:
            equipment_boxes[detected_classes[class_id]].append((box, score))
    person_count = 0
    people_with_equipment = 0
    people_without_equipment = 0

    for person in people:

        x1, y1, x2, y2,z= map(int, person)
        person_box = [x1, y1, x2, y2]
        detected_equipment = []  # Initialize a list for detected equipment
        for equip_type, boxes in equipment_boxes.items():
            for box, score in boxes:
                if are_bottom_points_within(box, person_box):
                    detected_equipment.append(equip_type.capitalize())
                    x1_equip, y1_equip, x2_equip, y2_equip = map(int, box)

        # Set label and color based on equipment detection
        if detected_equipment:
            people_with_equipment += 1
        else:
            people_without_equipment += 1
        person_count += 1
    if (people_with_equipment + people_without_equipment) == 0:
        total = 1
    else:
        total = people_with_equipment + people_without_equipment
    safety_index = int((people_with_equipment / total) * 100)
    data=[person_count,people_with_equipment,people_without_equipment,safety_index]
    return data



class VideoFeedConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.video_id = int(self.scope['url_route']['kwargs']['video_id'])

        await self.accept()
        # Open video capture for live streaming
        self.cap = cv2.VideoCapture(self.video_id)  # or replace with a video file path

        # Start the video streaming loop
        asyncio.create_task(self.stream_video())

    async def disconnect(self, close_code):
        # Close the video capture
        if self.cap.isOpened():
            self.cap.release()

    async def stream_video(self):
        
        global camera, results_data, person_count, people_with_equipment, people_without_equipment, previous_tracked_ids, start_time, elapsed_time
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            violator=None
            person_count = 0
            people_with_equipment = 0
            people_without_equipment = 0
            person_result = []
            # Run inference on the frame
            results = model(frame)
            detected_boxes = results[0].boxes.xyxy.cpu().numpy()
            detected_classes = results[0].names
            detected_scores = results[0].boxes.conf.cpu().numpy()
            detected_class_ids = results[0].boxes.cls.cpu().numpy()

            # Prepare detections for SORT (only include 'person' class)
            person_detections_for_sort = []
            for i, box in enumerate(detected_boxes):
                class_id = int(detected_class_ids[i])
                if detected_classes[class_id] == 'person' and detected_scores[i] > 0.50:
                    x1, y1, x2, y2 = box
                    score = detected_scores[i]
                    person_detections_for_sort.append([x1, y1, x2, y2, score])

            person_detections_for_sort = np.array(person_detections_for_sort) if person_detections_for_sort else np.empty((0, 5))

            # Update SORT tracker for persons
            tracked_persons = person_tracker.update(person_detections_for_sort)

            current_tracked_ids = set(int(person[4]) for person in tracked_persons)  # Get current person IDs

            # Detect disappeared IDs
            disappeared_ids = previous_tracked_ids - current_tracked_ids
            # for disappeared_id in disappeared_ids:
            #     # If a person disappears, adjust the counts accordingly
            #     if disappeared_id in person_equipment_data:
            #         if person_equipment_data[disappeared_id]['detected_equipment']:
            #             people_with_equipment -= 1 
            #         else:
            #             people_without_equipment -= 1 
            #         person_count -= 1

            # Update previous_tracked_ids for the next frame

            previous_tracked_ids = current_tracked_ids
            with open('app/compliances.json', 'r') as file:
                data = json.load(file)
            equipment_boxes = {compliance.lower(): [] for compliance in data['compliances']}


            # Classify and store equipment boxes
            for i, box in enumerate(detected_boxes):
                class_id = int(detected_class_ids[i])
                score = detected_scores[i]
                if detected_classes[class_id] in equipment_boxes and score > 0.30:
                    equipment_boxes[detected_classes[class_id]].append((box, score))
                    x1, y1, x2, y2 = map(int, box)
                    equipment_color = (255, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), equipment_color, 2)
                    cv2.putText(frame, detected_classes[class_id].capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, equipment_color, 2)

            # Draw tracked persons and check for equipment
            for person in tracked_persons:
                x1, y1, x2, y2, person_id = map(int, person)
                person_box = [x1, y1, x2, y2]

                if person_id not in person_frame_count:
                    person_frame_count[person_id] = 0
                    person_equipment_data[person_id] = {'detected_equipment': [], 'color': (247, 62, 237), 'notification_sent': False}

                person_frame_count[person_id] += 1

                if person_frame_count[person_id] == 20:
                    detected_equipment = []
                    for equip_type, boxes in equipment_boxes.items():
                        for box, score in boxes:
                            if are_bottom_points_within(box, person_box):
                                detected_equipment.append(equip_type.capitalize())
                    person_equipment_data[person_id]['detected_equipment'] = detected_equipment

                    # Update counts and flags based on equipment detection
                    if check_red_box(equipment_boxes,detected_equipment):
                        person_equipment_data[person_id]['color'] = (0, 255, 0)  # Green for adherence
                        person_count += 1  # Increment person count once
                        people_with_equipment += 1
                    else:
                        # Only send notification and change the color once
                        person_equipment_data[person_id]['color'] = (0, 0, 255)  # Set red bounding box
                        if not person_equipment_data[person_id]['notification_sent']:
                            violator = frame[y1:y2, x1:x2]
                            if violator.size == 0:
                                 print(f"Invalid violator region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            else:
                                person_count += 1
                                people_without_equipment += 1
                                results_data["violator"] = True
                                start_time = time.time()
                                contiguous_rgb_violator = np.ascontiguousarray(violator)
                                _, buffer = cv2.imencode('.jpg', contiguous_rgb_violator)  # Encode to JPG format
                                image_content = ContentFile(buffer.tobytes())
                                save_thread = threading.Thread(target=save_violator_instance, args=(person_id, image_content))
                                save_thread.start()
                                
                    person_frame_count[person_id] = 0

                # Draw bounding box and label for person
                color = person_equipment_data[person_id]['color']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                equipment_text = f"ID {person_id}: {', '.join(person_equipment_data[person_id]['detected_equipment'])}" if person_equipment_data[person_id]['detected_equipment'] else f"ID {person_id}: No equipment"
                cv2.putText(frame, equipment_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                person_result.append(f"id : {person_id} {', '.join(person_equipment_data[person_id]['detected_equipment']) if person_equipment_data[person_id]['detected_equipment'] else 'no equipment'}\n")
     


            # Calculate safety index
            total = people_with_equipment + people_without_equipment 
            if total == 0:
                total = 1 # Prevent division by zero
            safety=int((people_with_equipment / total) * 100)
            results_data['safety_index'] = {safety}
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            person_count,people_with_equipment,people_without_equipment,safety=get_analytics(results)
            analytics=f"Total People : {person_count} \n Adhering : {people_with_equipment} \n Violating : {people_without_equipment}"
            # if violator is None:
            #     message = {
            #         "frame": frame_data,
            #         "results": {
            #             "person_results": person_result,
            #             "analytics" : analytics,
            #             "safety_index": safety  # Ensure safety index is calculated
            #                 }
            #             }
            # else:
            #     _, buffer1 = cv2.imencode('.jpg', violator)
            #     violator_data = base64.b64encode(buffer1).decode('utf-8')
            #     print("violator")
            if violator is not None and violator.size != 0:
                  _, buffer1 = cv2.imencode('.jpg', violator)
                  violator_data = base64.b64encode(buffer1).decode('utf-8')
                  message = {
                    "frame": frame_data,
                    "violator" : violator_data,
                    "results": {
                    "person_results": person_result,
                    "analytics" : analytics,
                    "safety_index": safety  # Ensure safety index is calculated
                                }
                            }
            else:
                message = {
                    "frame": frame_data,
                    "results": {
                        "person_results": person_result,
                        "analytics" : analytics,
                        "safety_index": safety  # Ensure safety index is calculated
                            }
                        }  
                
            # Convert message to JSON
            message_json = json.dumps(message)

            # Send frame and result data to WebSocket
            await self.send(text_data=message_json)

            # Control frame rate
            await asyncio.sleep(0.03)  # Adjust as needed for 30 FPS

    async def receive(self, text_data):
        # Handle messages from frontend if needed
        pass





class UploadVideo(AsyncWebsocketConsumer):
     def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_chunks = []
        self.video_filename = None
        self.total_chunks = 0
        self.received_chunks = 0

     async def connect(self):
        await self.accept()
        print("WebSocket connected")

     async def disconnect(self, close_code):
        print(f"WebSocket disconnected with code: {close_code}")
        if self.video_chunks:
            print("Processing received video chunks")
            self.save_video()

     async def receive(self, text_data=None, bytes_data=None):
      if text_data:
        # Handle metadata for chunk information
        print("Received metadata:", text_data)
        message = json.loads(text_data)
        if message['type'] == 'video_chunk':
            file_name = message['file_name']
            chunk_index = message['chunk_index']
            total = message['total_chunks']
            print(f"Metadata for chunk {chunk_index + 1}/{total} received.")

            # Store file name and total chunks info
            self.video_filename = file_name
            self.total_chunks = total
            print(f"Expected total chunks: {self.total_chunks}")

            # Validate chunk index and ensure it is within the valid range
            if chunk_index < 0 or chunk_index >= total:
                print(f"Invalid chunk index {chunk_index}. Expected between 0 and {total - 1}.")
                return

            # Update the current chunk index
            self.current_chunk_index = chunk_index
            print(f"Valid chunk index: {chunk_index}")

      if bytes_data:
        # Handle binary video chunk data
        print(f"Received binary chunk, size: {len(bytes_data)} bytes")

        # Ensure the chunk index matches the metadata received earlier
        if hasattr(self, 'current_chunk_index') and self.current_chunk_index is not None:
            chunk_index = self.current_chunk_index
            print(f"Processing chunk {chunk_index + 1}/{self.total_chunks}")

            # Append binary data (video chunk) to the list
            self.video_chunks.append(bytes_data)
            print(f"Received chunks so far: {len(self.video_chunks)}")

            # Increment the received chunks counter
            self.received_chunks += 1

            # If all chunks have been received, proceed with saving the video
            if self.received_chunks == self.total_chunks:
                print("All chunks received, processing video...")
                await self.save_video()
        else:
            print(f"Error: No valid chunk index received for chunk {chunk_index + 1}.")

     async def save_video(self):
        # Check if all chunks are received
        if self.received_chunks == self.total_chunks:
            try:
                # Ensure video_chunks contains only byte data
                if not self.video_chunks:
                    print("Error: No video chunks received")
                    return

                # Print the total number of chunks to check the received chunks
                print(f"Received {len(self.video_chunks)} chunks of {self.total_chunks} total.")

                # Join the chunks into a single bytes object
                video_data = b''.join(self.video_chunks)
                print("Chunks combined")

                # Specify the file path to save the video
                video_path = f"upload_video/{self.video_filename}"
                print(f"Saving video to {video_path}")

                # Save the video to a file
                with open(video_path, 'wb') as video_file:
                    print("Writing video data to file...")
                    video_file.write(video_data)

                print(f"Video saved at {video_path}")

                # Proceed with further processing (e.g., inference)
                await self.stream_video(video_path)

            except TypeError as e:
                print(f"Error joining video chunks: {e}")
            except IOError as e:
                print(f"Error saving video file: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        else:
            print(f"Not all chunks received yet. Received {self.received_chunks}/{self.total_chunks} chunks.")


     async def stream_video(self,video_path):
        
        global camera, results_data, person_count, people_with_equipment, people_without_equipment, previous_tracked_ids, start_time, elapsed_time
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            violator=None
            person_count = 0
            people_with_equipment = 0
            people_without_equipment = 0
            person_result = []
            # Run inference on the frame
            results = model(frame)
            detected_boxes = results[0].boxes.xyxy.cpu().numpy()
            detected_classes = results[0].names
            detected_scores = results[0].boxes.conf.cpu().numpy()
            detected_class_ids = results[0].boxes.cls.cpu().numpy()

            # Prepare detections for SORT (only include 'person' class)
            person_detections_for_sort = []
            for i, box in enumerate(detected_boxes):
                class_id = int(detected_class_ids[i])
                if detected_classes[class_id] == 'person' and detected_scores[i] > 0.50:
                    x1, y1, x2, y2 = box
                    score = detected_scores[i]
                    person_detections_for_sort.append([x1, y1, x2, y2, score])

            person_detections_for_sort = np.array(person_detections_for_sort) if person_detections_for_sort else np.empty((0, 5))

            # Update SORT tracker for persons
            tracked_persons = person_tracker.update(person_detections_for_sort)

            current_tracked_ids = set(int(person[4]) for person in tracked_persons)  # Get current person IDs

            # Detect disappeared IDs
            disappeared_ids = previous_tracked_ids - current_tracked_ids
            # for disappeared_id in disappeared_ids:
            #     # If a person disappears, adjust the counts accordingly
            #     if disappeared_id in person_equipment_data:
            #         if person_equipment_data[disappeared_id]['detected_equipment']:
            #             people_with_equipment -= 1 
            #         else:
            #             people_without_equipment -= 1 
            #         person_count -= 1

            # # Update previous_tracked_ids for the next frame
            previous_tracked_ids = current_tracked_ids

            # Initialize equipment boxes
            # equipment_boxes = {'helmet': [], 'glasses': [], 'face-guard': [], 'gloves': [], 'shoes': [], 'safety-vest': []}
            with open('app/compliances.json', 'r') as file:
                data = json.load(file)
            equipment_boxes = {compliance.lower(): [] for compliance in data['compliances']}
            # Classify and store equipment boxes
            for i, box in enumerate(detected_boxes):
                class_id = int(detected_class_ids[i])
                score = detected_scores[i]
                if detected_classes[class_id] in equipment_boxes and score > 0.30:
                    equipment_boxes[detected_classes[class_id]].append((box, score))
                    x1, y1, x2, y2 = map(int, box)
                    equipment_color = (255, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), equipment_color, 2)
                    cv2.putText(frame, detected_classes[class_id].capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, equipment_color, 2)

            # Draw tracked persons and check for equipment
            for person in tracked_persons:
                x1, y1, x2, y2, person_id = map(int, person)
                person_box = [x1, y1, x2, y2]

                if person_id not in person_frame_count:
                    person_frame_count[person_id] = 0
                    person_equipment_data[person_id] = {'detected_equipment': [], 'color': (247, 62, 237), 'notification_sent': False}

                person_frame_count[person_id] += 1

                if person_frame_count[person_id] == 20:
                    detected_equipment = []
                    for equip_type, boxes in equipment_boxes.items():
                        for box, score in boxes:
                            if are_bottom_points_within(box, person_box):
                                detected_equipment.append(equip_type.capitalize())
                    person_equipment_data[person_id]['detected_equipment'] = detected_equipment
                    if check_red_box(equipment_boxes,detected_equipment):
                        person_equipment_data[person_id]['color'] = (0, 255, 0)  # Green for adherence
                        person_count += 1  # Increment person count once
                        people_with_equipment += 1
                    else:
                        # Only send notification and change the color once
                        person_equipment_data[person_id]['color'] = (0, 0, 255)  # Set red bounding box
                        if not person_equipment_data[person_id]['notification_sent']:
                            violator = frame[y1:y2, x1:x2]
                            if violator.size == 0:
                                 print(f"Invalid violator region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            else:
                                person_count += 1
                                people_without_equipment += 1
                                results_data["violator"] = True
                                start_time = time.time()
                                contiguous_rgb_violator = np.ascontiguousarray(violator)
                                _, buffer = cv2.imencode('.jpg', contiguous_rgb_violator)  # Encode to JPG format
                                image_content = ContentFile(buffer.tobytes())
                                save_thread = threading.Thread(target=save_violator_instance, args=(person_id, image_content))
                                save_thread.start()
                                
                    person_frame_count[person_id] = 0

                # Draw bounding box and label for person
                color = person_equipment_data[person_id]['color']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                equipment_text = f"ID {person_id}: {', '.join(person_equipment_data[person_id]['detected_equipment'])}" if person_equipment_data[person_id]['detected_equipment'] else f"ID {person_id}: No equipment"
                cv2.putText(frame, equipment_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                person_result.append(f"id : {person_id} {', '.join(person_equipment_data[person_id]['detected_equipment']) if person_equipment_data[person_id]['detected_equipment'] else 'no equipment'}\n")
     


            # Calculate safety index
            total = people_with_equipment + people_without_equipment 
            if total == 0:
                total = 1 # Prevent division by zero
            safety=int((people_with_equipment / total) * 100)
            results_data['safety_index'] = {safety}
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            person_count,people_with_equipment,people_without_equipment,safety=get_analytics(results)
            analytics=f"Total People : {person_count} \n Adhering : {people_with_equipment} \n Violating : {people_without_equipment}"
            if violator is None:
                message = {
                    "frame": frame_data,
                    "results": {
                        "person_results": person_result,
                        "analytics" : analytics,
                        "safety_index": safety  # Ensure safety index is calculated
                            }
                        }
            # else:
            #     _, buffer1 = cv2.imencode('.jpg', violator)
            #     violator_data = base64.b64encode(buffer1).decode('utf-8')
            #     print("violator")
            #     message = {
            #         "frame": frame_data,
            #         "violator" : violator_data,
            #         "results": {
            #         "person_results": person_result,
            #         "analytics" : analytics,
            #         "safety_index": safety  # Ensure safety index is calculated
            #                     }
            #                 }
                
                
            # Convert message to JSON
            message_json = json.dumps(message)

            # Send frame and result data to WebSocket
            await self.send(text_data=message_json)

            # Control frame rate
            await asyncio.sleep(0.03)  # Adjust as needed for 30 FPS
