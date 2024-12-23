from django.db import models
from django.utils import timezone

class People(models.Model):
    compliance = models.CharField(max_length=255, null=True)
    personid=models.CharField(max_length=225, null=True)
    image = models.ImageField(upload_to='violators/', blank=True, null=True)
    name=models.CharField(max_length=255, blank=True, null=True)
    action_taken = models.CharField(max_length=255, choices=[('Yes', 'Yes'), ('No', 'No')], default='No')
    identify_employee=models.CharField(max_length=255, choices=[('Yes', 'Yes'), ('No', 'No')], default='No')
    counsling=models.CharField(max_length=255, choices=[('Yes', 'Yes'), ('No', 'No')], default='No')
    reason = models.CharField(max_length=255, blank=True, null=True)
    created = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Violator {self.id} - Compliance: {self.compliance} - Created: {self.created}"


class QuizResponse(models.Model):
    name = models.CharField(max_length=100)
    id_number = models.CharField(max_length=20)
    question = models.TextField()
    selected_answer = models.CharField(max_length=10)
    correct_answer = models.CharField(max_length=10, default='')

    def __str__(self):
        return f"{self.name} - {self.question}"
    

class compliances(models.Model):
    name = models.CharField(max_length=255)  # Name of the compliance
    description = models.TextField(blank=True, null=True)  # Optional description
    is_active = models.BooleanField(default=True)  # Status of the compliance

    def __str__(self):
        return self.name
