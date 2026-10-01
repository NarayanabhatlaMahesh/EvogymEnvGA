import json
import os
import random
import re
import tempfile
from anyio import Path
from anyio import Path
import pandas as pd
import numpy as np
from evogym.world import EvoWorld
from .models import Generation, Individual, MetricsPerGeneration, MetricsPerRun, MetricsPerRun, World
from .JsonWorldEnv import JsonWorldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ollama import chat
from ollama import ChatResponse
from stable_baselines3 import PPO

class BaseGenerator:
    DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(
        self,
        world_w, world_h,
        obj_w, obj_h,
        min_voxels, max_voxels,
        num_objects
    ):
        self.world_w = world_w
        self.world_h = world_h
        self.obj_w = obj_w
        self.obj_h = obj_h
        self.min_voxels = min_voxels
        self.max_voxels = max_voxels
        self.num_objects = num_objects

    def gen_world(self):
        world, occupied = [], set()
        n = self.num_objects

        while len(world) < n:
            obj = self.gen_object()

            # Normalize object to origin (0, 0)
            minx = min(x for x, _ in obj)
            miny = min(y for _, y in obj)
            norm_obj = {(x - minx, y - miny) for x, y in obj}

            # Attempt placing the object in the world
            w = max(x for x, _ in norm_obj) + 1
            h = max(y for _, y in norm_obj) + 1
            placed = None

            for _ in range(100):
                ox = random.randint(0, self.world_w - w)
                oy = random.randint(0, self.world_h - h)
                translated = {(x + ox, y + oy) for x, y in norm_obj}

                if not (translated & occupied):
                    placed = translated
                    break

            if placed is None:
                continue

            occupied.update(placed)
            
            # Format JSON output inline
            obj_list = list(placed)
            lookup = {p: i for i, p in enumerate(obj_list)}
            world.append({
                "indices": self.get_indices(obj_list),
                "types": [5] * len(obj_list),
                "neighbors": [
                    [lookup[n] for n in self.get_neighbors(v) if n in lookup]
                    for v in obj_list
                ],
            })

        return world

    def gen_object(self):
        target = random.randint(self.min_voxels, self.max_voxels)
        obj = {(random.randrange(self.obj_w), random.randrange(self.obj_h))}

        while len(obj) < target:
            frontier = [
                n
                for voxel in obj
                for n in self.get_neighbors(voxel)
                if 0 <= n[0] < self.obj_w and 0 <= n[1] < self.obj_h and n not in obj
            ]
            if not frontier:
                break
            obj.add(random.choice(frontier))

        return obj

    def get_neighbors(self, voxel):
        x, y = voxel
        return [(x + dx, y + dy) for dx, dy in self.DIRS]

    def get_indices(self, obj_list):
        return [[x, y] for x, y in obj_list]

    def get_island(self, start_voxel, obj):
        """BFS traversal to retrieve a connected cluster (island) of voxels."""
        q, seen = [start_voxel], set()
        while q:
            v = q.pop(0)
            if v not in seen:
                seen.add(v)
                q.extend([n for n in self.get_neighbors(v) if n in obj and n not in seen])
        return seen



class VoxelMutator:
    DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self, default_type: int = 5):
        self.default_type = default_type

    def get_neighbors(self, voxel: tuple) -> list:
        """Returns 4-directional 2D grid neighbors of a voxel (x, y)."""
        x, y = voxel
        return [(x + dx, y + dy) for dx, dy in self.DIRS]

    def get_outer(self, obj_json: dict) -> list:
        """Finds empty neighbor coordinates adjacent to the object (outer growth boundary)."""
        existing_indices = set(tuple(p) for p in obj_json["indices"])
        outer = set()

        for voxel in existing_indices:
            for n in self.get_neighbors(voxel):
                if n not in existing_indices:
                    outer.add(n)

        return list(outer)

    def get_perimeter(self, obj_json: dict) -> list:
        """Finds existing voxels on the boundary (voxels with at least 1 empty neighbor)."""
        existing_indices = set(tuple(p) for p in obj_json["indices"])
        perimeter = []

        for voxel in obj_json["indices"]:
            v_tuple = tuple(voxel)
            # A voxel is on the perimeter if any of its 4 directions is empty
            if any(n not in existing_indices for n in self.get_neighbors(v_tuple)):
                perimeter.append(v_tuple)

        return perimeter

    def add_voxel(self, obj_json: dict, target_voxel: tuple = None) -> bool:
        """Adds a new voxel at the outer boundary and updates neighbor adjacency lists."""
        outer = self.get_outer(obj_json)
        if not outer:
            return False

        # Use passed target (e.g., selected by LLM/Agent) or pick randomly
        if target_voxel is None:
            target_voxel = random.choice(outer)
        elif target_voxel not in outer:
            return False  # Target position is invalid for growth

        target_list = list(target_voxel)
        new_idx = len(obj_json["indices"])

        # 1. Append new voxel metadata
        obj_json["indices"].append(target_list)
        obj_json["types"].append(self.default_type)
        obj_json["neighbors"].append([])

        # 2. Update bidirectional adjacency connections
        for idx, voxel in enumerate(obj_json["indices"][:-1]):
            if tuple(voxel) in self.get_neighbors(target_voxel):
                obj_json["neighbors"][new_idx].append(idx)
                obj_json["neighbors"][idx].append(new_idx)

        return True

    def delete_voxel(self, obj_json: dict, target_voxel: tuple = None) -> bool:
        """Removes a perimeter voxel and re-indexes all neighbor relationships."""
        perimeter = self.get_perimeter(obj_json)
        if not perimeter or len(obj_json["indices"]) <= 1:
            return False  # Avoid removing the last voxel

        # Select target perimeter voxel
        if target_voxel is None:
            target_voxel = random.choice(perimeter)
        elif target_voxel not in perimeter:
            return False

        target_list = list(target_voxel)
        remove_idx = obj_json["indices"].index(target_list)

        # 1. Remove voxel entries from indices, types, and neighbors list
        obj_json["indices"].pop(remove_idx)
        obj_json["types"].pop(remove_idx)
        obj_json["neighbors"].pop(remove_idx)

        # 2. Shift neighbor references to adjust for index changes
        for neighbor_list in obj_json["neighbors"]:
            if remove_idx in neighbor_list:
                neighbor_list.remove(remove_idx)

            # Re-index remaining neighbor IDs higher than remove_idx
            for i in range(len(neighbor_list)):
                if neighbor_list[i] > remove_idx:
                    neighbor_list[i] -= 1

        return True

    def mutate(self, obj_json: dict, action: str = "add", target_voxel: tuple = None) -> bool:
        """High-level entry point to apply an 'add' or 'delete' mutation step."""
        if action == "add":
            return self.add_voxel(obj_json, target_voxel)
        elif action in ("delete", "subtract"):
            return self.delete_voxel(obj_json, target_voxel)
        else:
            raise ValueError(f"Unknown action: {action}")
        


import math
import random


class EvolveExperiment:

    def __init__(
        self,
        ground_width,
        world_height,
        timesteps_per_run,
        seed,
        gen_count,
        population_size,
        mutation_rate,
        elitism_count,
        experiment,
        selection_strategy,
        max_voxel_width=5,
        obj_w=6,
        obj_h=6,voxel_count_min=5, voxel_count_max=12, obj_count=3
    ):
        self.ground_width = ground_width
        self.world_height = world_height
        self.timesteps_per_run = timesteps_per_run
        self.seed = seed
        self.gen_count = gen_count
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.selection_strategy = selection_strategy
        self.max_voxel_width = max_voxel_width
        self.ROBOT_NAME = "roboname"
        self.base_path = Path(
            r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\RunExp\TrainingFiles\FilterRobots\Trained_Models_Sample"
        )
        self.csv_path = (r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\EvoPlayground\TraverseLab\Experiments\Utils\excluded_encs.csv")
        self.obj_w=obj_w
        self.obj_h=obj_h
        self.voxel_count_min=voxel_count_min
        self.voxel_count_max=voxel_count_max
        self.obj_count=obj_count    
        self.model_map = self._load_models()
        self.experiment = experiment

    def parse_array_blocks(self, text):
        arrays = re.findall(r'array\(\[([^\]]+)\]\)', text)
        return np.array([np.fromstring(a, sep=',', dtype=int) for a in arrays])


    def load_robot_from_csv(self, csv_path, env_name, row_idx):
        print(f"Loading robot from {csv_path}")
        df = pd.read_csv(csv_path)
        df = df.where(df['env_name'] == 'ObstacleTraverser-v0').dropna().sort_values(by='reward').reset_index(drop=True)
        print(f"Found {len(df)} robots for environment {env_name}, loading row {row_idx}")
        row = df.iloc[row_idx]
        body = self.parse_array_blocks(row["body"])
        connections = self.parse_array_blocks(row["connections"])
        return body, connections

    def _load_models(self):
        df = pd.read_csv(self.csv_path).reset_index(drop=True)
        model_map = {}
        for i in range(len(df)):
            robo_dir = self.base_path / f"Robo{i}"
            if not robo_dir.exists():
                continue
            model_files = list(robo_dir.glob("*.zip"))
            if not model_files:
                continue
            try:
                body = self.parse_array_blocks(df.iloc[i]["body"])
                connections = self.parse_array_blocks(df.iloc[i]["connections"])
            except Exception as e:
                print(f"CSV parse failed for Robo{i}: {e}")
                continue
            for model_path in model_files:
                try:
                    model_map[str(model_path)] = {
                        "model"         : PPO.load(str(model_path),device="cpu"),
                        "body"          : body,
                        "connections"   : connections,
                        "env_class"     : JsonWorldEnv,
                        "model_path"    : str(model_path),
                    }
                except Exception as e:
                    print(f"Skipping {model_path}: {e}")
                    continue
        return model_map
    
    def get_robot_set(self):
        """returns list of models loaded PPOs"""
        models = {}
        for i in range(len(pd.read_csv(self.csv_path))):
            models[i] = PPO.load(self.base_path / f"Robo{i}")
        return models

    def get_base_child(self, parent1, parent2):
        """Recombines Parent 1 and Parent 2 objects into a child with 1-voxel gaps and applies scaled mutations."""
        p1_objs = parent1["objects"]
        p2_objs = parent2["objects"]
        child_objs = []
        switch = True
        num_objs = math.floor( (self.ground_width + 1) / (self.max_voxel_width + 1))
        n1, n2 = 0, 0
        while True:
            if switch:
                if n1 < len(p1_objs):
                    child_objs.append(p1_objs[n1])
                    n1 += 1
            else:
                if n2 < len(p2_objs):
                    child_objs.append(p2_objs[n2])
                    n2 += 1
            switch = not switch  # Toggle parent source
            if (len(child_objs) == num_objs or (n1 >= len(p1_objs) and n2 >= len(p2_objs))):
                break
        mp = int(self.mutation_rate * 10)
        while mp > 0:
            for obj in child_objs:
                r = random.choice([0, 1])
                self.mutate(obj, r, not r)
            mp -= 1
        return {"grid_width": self.ground_width, "grid_height": self.world_height, "objects": child_objs,}

    def mutate(self, obj, r1, r2):
        """Placeholder for your VoxelMutator / LLM selector logic."""
        vm = VoxelMutator()
        vm.mutate(obj, action="add" if r1 else "delete")
        pass

    def run_evolution(self):
        """Main Evolutionary Loop."""
        random.seed(self.seed)
        gen_candies = [ BaseGenerator(world_w=self.ground_width, world_h=self.world_height, obj_w=self.obj_w,
                obj_h=self.obj_h, min_voxels=self.voxel_count_min, max_voxels=self.voxel_count_max, num_objects=self.obj_count
                ).gen_world() for _ in range(self.population_size) ]
        for gen in range(self.gen_count):
            print("candies generated for gen", g)
            generation = Generation(experiment=self.experiment, generation_number=gen).save()
            v = self.run_evogym_evaluator(gen_candies, generation=generation)
            v.sort(key=lambda x: x["fitness"], reverse=True)
            MetricsPerGeneration.objects.create(experiment=self.experiment, generation=generation, 
                                                avg_fitness=np.mean([x["avgfitness"] for x in v]), best_fitness=np.max([x["best_fitness"] for x in v]),
                                                avg_objects_crossed=np.mean([x["avg_objects_crossed"] for x in v]), avg_distance_travelled=np.mean([x["avg_distance_travelled"] for x in v]))
            parents = v[: self.elitism_count]
            next_gen = [p["world_data"] for p in parents]
            for i in range(len(parents)):
                for j in range(len(parents)):
                    if i != j:
                        child = self.get_base_child( parents[i]["world_data"], parents[j]["world_data"] )
                        next_gen.append(child)
                        if len(next_gen) >= self.population_size:
                            break
                if len(next_gen) >= self.population_size:
                    break
            gen_candies = next_gen[: self.population_size]
        return gen_candies
    
    def run_evogym_evaluator(self, worlds, generation):
        evaluated = []
        for world_desc in worlds:
            world_json = self.to_object_json(world_desc)
            World.objects.create(generation=generation, world_json=world_json)
            with tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False) as f:
                json.dump(world_json, f)
                json_path = f.name
            try:
                fitness_scores = []
                robot_data = []
                robos = Individual.objects.all()
                for job in robos:
                    print(f"Evaluating robot {job.robo_id} on world {json_path}")
                    world = EvoWorld.from_json(json_path)
                    
                    world.add_from_array( name=self.ROBOT_NAME, structure=job.genome_data["body"], connections=job.genome_data["connections"],
                        x=1, y=10, )
                    World.objects.create( generation=generation, world_json=world_json)
                    env = JsonWorldEnv( world=world, robot_name=self.ROBOT_NAME, total_timesteps=self.timesteps_per_run,)
                    obs, _ = env.reset()
                    start_x = float( env.get_pos_com_obs(self.ROBOT_NAME)[0] )
                    max_x = start_x
                    runs = self.timesteps_per_run
                    while (runs>0):
                        runs -= 1
                        action, _ = job.model_path.predict( obs, deterministic=True )
                        obs, reward, terminated, truncated, _ = env.step(action)
                        cur_x = float(env.get_pos_com_obs(self.ROBOT_NAME)[0])
                        if cur_x > max_x:
                            max_x = cur_x
                        if terminated or truncated:
                            break
                    env.close()
                    distance = max_x - start_x
                    score = distance / self.ground_width
                    fitness_scores.append(score)
                    robot_data.append({
                        "model_path": job.model_path,
                        "distance": distance,
                        "score": score,
                    })
                    num_objs_crossed = self.get_num_objs_crossed(distance, world_json)
                    print(f"Robot {job.model_path} crossed {len(num_objs_crossed)} objects in world {json_path}")
                    MetricsPerRun.objects.create(individual=job,
                        world=World.objects.get(generation=generation, world_json=world_json),
                        run_number=1,
                        objects_crossed=len(num_objs_crossed),
                        distance_travelled=distance, calculated_fitness=score)
                fitness = float(np.mean(fitness_scores))
                evaluated.append({ "world_data": world_desc,
                    "avgfitness": fitness, "robots": robot_data, 'fitness_scores': fitness_scores, "best_fitness": np.max(fitness_scores), 
                    "avg_objects_crossed": np.mean([x["objects_crossed"] for x in robot_data]), "avg_distance_travelled": np.mean([x["distance"] for x in robot_data]) })
            finally:
                os.unlink(json_path)
        return evaluated

    def get_num_objs_crossed(self, travelled, json):
        width = self.ground_width
        crossed=[]
        for keys, vals in json['objects'].items():
            max=0
            for i in json['objects'][keys]['indices']:
                if (i < width) and (i > max):
                    max = i
                    crossed.append(json['objects'][keys])
            if max < travelled:
                crossed.append(json['objects'][keys])
        return crossed

class Train():
    def __init__(self, ROBOT_CSV, ROBOT_NAME, PPO_TRAIN_TIMESTEPS, N_ENVS, N_STEPS, BATCH_SIZE, DEVICE, MAX_STEPS):
        self.ROBOT_CSV = ROBOT_CSV
        self.ROBOT_NAME = ROBOT_NAME
        self.PPO_TRAIN_TIMESTEPS = PPO_TRAIN_TIMESTEPS
        self.N_ENVS = N_ENVS
        self.N_STEPS = N_STEPS
        self.BATCH_SIZE = BATCH_SIZE
        self.DEVICE = DEVICE
        self.MAX_STEPS = MAX_STEPS

    # ---------------- FITNESS ----------------
    def ppo_fitness(self, env_desc):
        world_json = self.to_object_json(env_desc)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(world_json, f)
            path = f.name

        try:
            world = EvoWorld.from_json(path)

            df = pd.read_csv(self.ROBOT_CSV)
            row = df.iloc[1]
            body = self.parse_body(row["body"])
            connections = self.parse_connections(row["connections"])

            world.add_from_array(
                name=self.ROBOT_NAME,
                structure=body,
                connections=connections,
                x=1,
                y=10,
            )

            env = DummyVecEnv([lambda: JsonWorldEnv(world=world, robot_name=self.ROBOT_NAME, total_timesteps=self.PPO_TRAIN_TIMESTEPS ) for _ in range(self.N_ENVS)])
            model = PPO( "MlpPolicy",
                env, n_steps=self.N_STEPS, batch_size=self.BATCH_SIZE, learning_rate=1e-3, device=self.DEVICE, verbose=0, )
            model.learn(total_timesteps=self.PPO_TRAIN_TIMESTEPS)
            obs = env.reset()
            total_reward = 0.0
            for _ in range(self.MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, _ = env.step(action)
                total_reward += reward.mean()
                if dones.any():
                    break
            env.close()
            return total_reward
        finally:
            os.unlink(path)
    
    def to_object_json(self, world):
        objects = {}
        ground_indices = list(range(self.ground_width))
        ground_neighbors = {}
        for i in ground_indices:
            nbrs = []
            if i > 0:
                nbrs.append(i - 1)
            if i < self.ground_width - 1:
                nbrs.append(i + 1)
            ground_neighbors[str(i)] = nbrs
        objects["ground"] = {
            "indices": ground_indices,
            "types": [1] * len(ground_indices),
            "neighbors": ground_neighbors,
        }
        for obj_id, obj in enumerate(world):
            coord_to_idx = {}
            flat_indices = []
            for x, y in obj["indices"]:
                idx = y * self.ground_width + x
                coord_to_idx[(x, y)] = idx
                flat_indices.append(idx)
            neighbors = {}
            for i, nbrs in enumerate(obj["neighbors"]):
                voxel = tuple(obj["indices"][i])
                neighbors[str(coord_to_idx[voxel])] = [
                    coord_to_idx[tuple(obj["indices"][j])]
                    for j in nbrs
                ]
            objects[f"terrain_{obj_id}"] = {
                "indices": flat_indices,
                "types": obj["types"],
                "neighbors": neighbors,
            }
        return {
            "grid_width": self.ground_width,
            "grid_height": self.world_height,
            "objects": objects,
        }
    
    def parse_body(self,body_str):
        arrays = re.findall(r'array\(\[([^\]]+)\]\)', body_str)
        return np.array([
            np.fromstring(a, sep=',', dtype=int)
            for a in arrays
        ])

    def parse_connections(self,conn_str):
        arrays = re.findall(r'array\(\[([^\]]+)\]\)', conn_str)
        return np.array([
            np.fromstring(a, sep=',', dtype=int)
            for a in arrays
        ])