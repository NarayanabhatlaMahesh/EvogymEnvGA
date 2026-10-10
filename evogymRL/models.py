from django.core.validators import FileExtensionValidator
from django.db import models

# Create your models here.
class PPOModels(models.Model):
    zipfile = models.FileField(upload_to="PPO_Files/", validators=[FileExtensionValidator(allowed_extensions=["Zip"])],)
    timestep = models.BigIntegerField()

class Reward(models.Model):
    func_choices = (('DX', 'dx'), ('DY', 'dy'), ('COM', 'com'), ('ANGLE', 'angle'))
    oper_choices = (('+', '+'), ('-', '-'), ('*', '*'))
    functions= models.CharField(max_length=255,choices=func_choices)
    operation = models.CharField(max_length=255, choices=oper_choices)
    value = models.IntegerField()
    def get_func_choices(self):
        return self.func_choices
    def get_oper_choices(self):
        return self.oper_choices    