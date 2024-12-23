# views.py
from django.shortcuts import render,redirect
from .models import QuizResponse,People
from django.views.decorators.csrf import csrf_protect
import openpyxl
from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse
import json
import torch  # Assuming you're using PyTorch for YOLO model loading

import os
def select_compliances(request):
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            data = json.loads(request.body)
            compliances = data.get('compliances', [])
            equipment_boxes = {compliance.lower(): [] for compliance in data['compliances']}

            # Define the path for the JSON file
            file_path = os.path.join(os.path.dirname(__file__), 'compliances.json')

            # Write the data to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump({'compliances': compliances}, json_file, indent=4)

            # Send a success response
            return JsonResponse({'message': 'Compliances saved successfully!'}, status=200)

        except Exception as e:
            # Handle errors and send a failure response
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    

def video(request):
    return render(request, 'app/video_feed.html')




def quiz_view(request):
    return render(request,'app/quiz.html')



#violator image html
def violator_image(request):
    violators = People.objects.all()
    return render(request, "app/violator_image.html", {'violators': violators})


def violator_list(request):
    if request.method == "POST":
        # Process the form submission
        for violator in People.objects.all():
            if f'save_{violator.id}' in request.POST:  # Check which save button was clicked
                name=request.POST.get(f'name_{violator.id}')
                action_taken = request.POST.get(f'action_taken_{violator.id}')
                reason = request.POST.get(f'reason_{violator.id}')
                identify_employee = request.POST.get(f'identify_employee_{violator.id}')
                counsling = request.POST.get(f'counsling_{violator.id}')
                
                # Update the violator's details
                violator.name=name
                violator.action_taken = action_taken
                violator.reason = reason
                violator.identify_employee = identify_employee
                violator.counsling = counsling
                violator.save()
        
        return redirect('violator_list')  # Redirect to avoid form resubmission issues

    # Fetch the violators to display
    violators = People.objects.all()
    return render(request, 'app/record.html', {'violators': violators})


def violator_report(request):
    violators = People.objects.all()  # Fetch all violators
    return render(request, 'app/report.html', {'violators': violators})


def export_to_excel(request):
    # Query violator data
    violators = People.objects.all()

    # Create a new workbook and select the active sheet
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Violator Report"

    # Define the column headers
    headers = ["Name", "Action Taken", "Identify Employee", "Counseling", "Reason"]
    ws.append(headers)

    # Add violator data to the Excel sheet
    for violator in violators:
        row = [
              # Include image URL or leave blank
            violator.name,
            violator.action_taken,
            violator.identify_employee,
            violator.counsling,
            violator.reason,
        ]
        ws.append(row)

    # Set the response content type and disposition for download
    response = HttpResponse(
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response["Content-Disposition"] = "attachment; filename=violator_report.xlsx"

    # Save the workbook to the response
    wb.save(response)
    return response





