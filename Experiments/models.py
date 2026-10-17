import uuid
from django.db import models


class Experiment(models.Model):
    name = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Environment Hyperparameters
    ground_width = models.IntegerField()
    world_height = models.IntegerField()
    timesteps_per_run = models.IntegerField()

    # Reproducibility
    seed = models.IntegerField(default=42)

    # GA Hyperparameters
    population_size = models.IntegerField(default=50)
    mutation_rate = models.FloatField()
    crossover_rate = models.FloatField(default=0.7)
    elitism_count = models.IntegerField(default=2)
    selection_strategy = models.CharField(max_length=50, default="tournament")

    def __str__(self):
        return str(self.name)


class Generation(models.Model):
    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, related_name="generations"
    )
    generation_number = models.IntegerField()

    def __str__(self):
        return f"Gen {self.generation_number} (Exp: {self.experiment.name})"


class Individual(models.Model):
    robo_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    # Morphological Graph JSON
    body = models.TextField()
    connections = models.TextField()
    model_path = models.CharField(max_length=255)

class World(models.Model):
    """Environment layout generated for a generation or specific trial."""

    generation = models.ForeignKey(
        Generation, on_delete=models.CASCADE, related_name="worlds"
    )
    world_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    world_json = models.TextField()

    def __str__(self):
        return f"World {self.world_id} - Gen {self.generation.generation_number}"


class MetricsPerRun(models.Model):
    """Evaluation trial metrics connecting an Individual robot to a World."""

    individual = models.ForeignKey(
        Individual, on_delete=models.CASCADE, related_name="run_metrics"
    )
    world = models.ForeignKey(
        World, on_delete=models.CASCADE, related_name="run_metrics"
    )
    run_number = models.IntegerField()
    objects_crossed = models.IntegerField() 
    distance_travelled = models.FloatField()
    calculated_fitness = models.FloatField()

    def __str__(self):
        return f"Run {self.run_number} for Robot {self.individual.robo_id}"


class MetricsPerGeneration(models.Model):
    """Aggregate statistics for an entire generation."""

    generation = models.ForeignKey(
        Generation, on_delete=models.CASCADE, related_name="gen_metrics"
    )
    best_fitness = models.FloatField()
    avg_fitness = models.FloatField()
    avg_objects_crossed = models.FloatField()
    avg_distance_travelled = models.FloatField()

    def __str__(self):
        return f"Gen {self.generation.generation_number} Aggregated Summary"

class ExperimentChartPoint(models.Model):
    experiment = models.ForeignKey(
        Experiment,
        on_delete=models.CASCADE,
        related_name="chart_points",
    )
    data_point_num = models.PositiveSmallIntegerField()
    generation_start = models.IntegerField()
    generation_end = models.IntegerField()

    best_fitness = models.FloatField()
    avg_fitness = models.FloatField()
    avg_objects_crossed = models.FloatField()
    avg_distance_travelled = models.FloatField()

    class Meta:
        ordering = ["data_point_num"]
        constraints = [
            models.UniqueConstraint(
                fields=["experiment", "data_point_num"],
                name="unique_experiment_chart_point",
            )
        ]