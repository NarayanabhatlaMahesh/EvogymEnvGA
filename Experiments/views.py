from django.db.models import Prefetch
from django.shortcuts import render
from Experiments.serializers.serializers import ExperimentSerializer
from Experiments.models import Experiment, ExperimentChartPoint, MetricsPerGeneration
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from .Env import EvolveExperiment

# Create your views here.
@api_view(["POST"])
@permission_classes([AllowAny])
def run_experiment(request):

    serializer = ExperimentSerializer(data=request.data)

    if not serializer.is_valid():
        print("Validation errors:", serializer.errors)
        print("Received data:", request.data)
        return Response(serializer.errors, status=400)
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




@api_view(["GET"])
@permission_classes([AllowAny])
def get_experiments(request):
    exps = Experiment.objects.prefetch_related(
        Prefetch(
            "chart_points",
            queryset=ExperimentChartPoint.objects.order_by("data_point_num"),
            to_attr="dashboard_points",
        )
    )

    result = []

    for exp in exps:
        result.append({
            "id": exp.id,
            "name": str(exp.name),
            "description": exp.description,
            "created_at": exp.created_at,
            "chart_points": [
                {
                    "data_point_num": p.data_point_num,
                    "best_fitness": p.best_fitness,
                    "avg_fitness": p.avg_fitness,
                    "avg_objects_crossed": p.avg_objects_crossed,
                    "avg_distance_travelled": p.avg_distance_travelled,
                }
                for p in exp.dashboard_points
            ],
        })

    return Response({"experiments": result})
@api_view(["GET"])
@permission_classes([AllowAny])
def get_Dashboard_matrics(request):
    exps = Experiment.objects.all()
    metrics = []
    for i in exps:
        metric = ExperimentChartPoint.objects.filter(experiment=i).values()
        metrics.extend(metric)

    return Response({"metrics": metrics}) 


@api_view(["POST"])
@permission_classes([AllowAny])
def get_Dashboard_matrics(request):
    exps = Experiment.objects.all()
    experimentuid = request.data.get("experimentuid")
    if experimentuid:
        exps = exps.filter(id=experimentuid)
    metrics = []
    for i in exps:
        metric = ExperimentChartPoint.objects.filter(experiment=i).values()
        metrics.extend(metric)
    return Response({"metrics": metrics}) 