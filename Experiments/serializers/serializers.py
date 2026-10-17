from rest_framework import serializers


class ExperimentSerializer(serializers.Serializer):

    ground_width = serializers.IntegerField(min_value=1)
    world_height = serializers.IntegerField(min_value=1)
    timesteps_per_run = serializers.IntegerField(min_value=1)
    seed = serializers.IntegerField()

    gen_count = serializers.IntegerField(min_value=1)

    population_size = serializers.IntegerField(min_value=1)
    mutation_rate = serializers.FloatField(min_value=0, max_value=1)
    crossover_rate = serializers.FloatField(min_value=0, max_value=1)
    elitism_count = serializers.IntegerField(min_value=0)
    selection_strategy = serializers.ChoiceField(
        choices=["tournament", "roulette", "rank"]
    )

    obj_w = serializers.IntegerField(min_value=1)
    obj_h = serializers.IntegerField(min_value=1)

    voxel_count_min = serializers.IntegerField(min_value=1)
    voxel_count_max = serializers.IntegerField(min_value=1)

    obj_count = serializers.IntegerField(min_value=1)

    def validate(self, attrs):
        if attrs["voxel_count_min"] > attrs["voxel_count_max"]:
            raise serializers.ValidationError({
                "voxel_count_min": "Minimum voxels cannot exceed maximum voxels."
            })

        if attrs["elitism_count"] > attrs["population_size"]:
            raise serializers.ValidationError({
                "elitism_count": "Elitism cannot exceed population size."
            })

        return attrs