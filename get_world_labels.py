#!/usr/bin/env python3
import os
import glob
import re
import json
import yaml

def collect_labels(base_dir):
    # initialize groups
    groups = {
        "difficulty":    {"low": [], "mid": [], "high": []},
        "obstacle_density": {"sparse": [], "medium": [], "dense": []},
        "terrain_type":  {"rigid": [], "deformable": [], "mixed": []}
    }

    pattern = os.path.join(base_dir, "config*_*.yaml")
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        m = re.match(r"config(\d+)_", fname, re.IGNORECASE)
        if not m:
            continue
        world_id = int(m.group(1))

        # load YAML
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        # difficulty from terrain.difficulty
        diff = cfg.get("terrain", {}).get("difficulty")
        if diff in groups["difficulty"]:
            groups["difficulty"][diff].append(world_id)
        else:
            groups["difficulty"].setdefault(diff, []).append(world_id)

        # obstacle_density
        od = cfg.get("obstacle_density")
        if od in groups["obstacle_density"]:
            groups["obstacle_density"][od].append(world_id)
        else:
            groups["obstacle_density"].setdefault(od, []).append(world_id)

        # terrain_type
        tt = cfg.get("terrain_type")
        if tt in groups["terrain_type"]:
            groups["terrain_type"][tt].append(world_id)
        else:
            groups["terrain_type"].setdefault(tt, []).append(world_id)

    # sort each list
    for category in groups.values():
        for label, lst in category.items():
            lst.sort()

    return groups

def main():
    BASE_DIR = "/home/zkr/Documents/verti_bench/envs/data/BenchMaps/sampled_maps/Configs/Final"
    labels = collect_labels(BASE_DIR)
    out_path = os.path.join(BASE_DIR, "config_labels.json")
    with open(out_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Written grouped labels to {out_path}")

if __name__ == "__main__":
    main()
