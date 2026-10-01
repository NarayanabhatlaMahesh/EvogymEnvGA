from django.shortcuts import render
from Experiments.serializers.serializers import ExperimentSerializer
from Experiments.models import Experiment
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from .Env import EvolveExperiment

# Create your views here.
@api_view(["POST"])
@permission_classes([AllowAny])
def run_experiment(request):

    serializer = ExperimentSerializer(data=request.data)

    serializer.is_valid(raise_exception=True)

    data = serializer.validated_data
    expobj = Experiment.objects.create(
        ground_width=data["ground_width"],
        world_height=data["world_height"],
        timesteps_per_run=data["timesteps_per_run"],
        seed=data["seed"],
        population_size=data["population_size"],
        mutation_rate=data["mutation_rate"],
        crossover_rate=data["crossover_rate"],
        elitism_count=data["elitism_count"],
        selection_strategy=data["selection_strategy"],
    )

    env = EvolveExperiment(
        ground_width=data["ground_width"],
        world_height=data["world_height"],
        timesteps_per_run=data["timesteps_per_run"],
        seed=data["seed"],
        gen_count=data["gen_count"],
        population_size=data["population_size"],
        mutation_rate=data["mutation_rate"],
        elitism_count=data["elitism_count"],
        selection_strategy=data["selection_strategy"],
        obj_w=data["obj_w"],
        obj_h=data["obj_h"],
        voxel_count_min=data["voxel_count_min"],
        voxel_count_max=data["voxel_count_max"],
        obj_count=data["obj_count"],
        experiment=expobj
    )

    world = env.run_evolution()

    return Response({
        "world": world
    })