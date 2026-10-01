from rest_framework import serializers


class ExperimentSerializer(serializers.Serializer):

    ground_width = serializers.IntegerField()
    world_height = serializers.IntegerField()
    timesteps_per_run = serializers.IntegerField()
    seed = serializers.IntegerField()

    gen_count = serializers.IntegerField()

    population_size = serializers.IntegerField()
    mutation_rate = serializers.FloatField()
    crossover_rate = serializers.FloatField()
    elitism_count = serializers.IntegerField()
    selection_strategy = serializers.CharField()

    obj_w = serializers.IntegerField()
    obj_h = serializers.IntegerField()

    voxel_count_min = serializers.IntegerField()
    voxel_count_max = serializers.IntegerField()

    obj_count = serializers.IntegerField()