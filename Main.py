"""
E-NSGA-II for Docker-based Test Optimization
Author: Ali Ardeshiri
"""

import random
import numpy as np
import subprocess
import json
import time
from typing import List, Tuple
import docker
from dataclasses import dataclass

# ==================== 1. Configuration & Data Structures ====================
@dataclass
class TestCase:
    id: int
    name: str
    estimated_duration: float  # Estimated execution time on baseline resource
    cpu_intensity: float       # 0.0 to 1.0

@dataclass
class DockerResource:
    id: int
    name: str          # e.g., "small", "medium", "large"
    cpu_limit: float   # e.g., 1.0, 2.0, 4.0
    power_model: float # Watts per CPU core (idle + dynamic)

@dataclass
class Chromosome:
    """Encodes a test configuration"""
    selection: List[bool]          # Which tests are selected
    ordering: List[int]            # Order of selected test indices
    resource_assignment: List[int] # Resource ID for each selected test
    
    def decode(self) -> List[Tuple[TestCase, DockerResource]]:
        """Decodes chromosome to executable plan"""
        # This would map to actual test and resource objects
        pass

# ==================== 2. Benchmark Systems & Resources ====================
# Define benchmark systems
benchmarks = {
    "commons-math": {
        "docker_image": "openjdk:11-jdk-slim",
        "test_command": "mvn test -Dtest=**/*Test",
        "test_cases": [TestCase(i, f"Test{i}", random.uniform(1, 10), random.random()) 
                      for i in range(100)]  # Simulate 100 tests
    },
    "jfreechart": {
        "docker_image": "openjdk:11-jdk-slim",
        "test_command": "./gradlew test",
        "test_cases": [TestCase(i, f"Test{i}", random.uniform(0.5, 8), random.random()) 
                      for i in range(80)]
    }
}

# Define cloud resources (Docker container configurations)
resources = [
    DockerResource(0, "small", 1.0, 12.0),   # 12W per container
    DockerResource(1, "medium", 2.0, 25.0),  # 25W per container
    DockerResource(2, "large", 4.0, 45.0)    # 45W per container
]

# ==================== 3. Docker Controller ====================
class DockerTestRunner:
    """Manages test execution in Docker containers"""
    
    def __init__(self):
        self.client = docker.from_env()
        
    def run_test_in_container(self, test: TestCase, resource: DockerResource, 
                              benchmark: str) -> dict:
        """
        Executes a single test in a Docker container with resource limits
        Returns: actual_time, energy_estimate, coverage
        """
        container = None
        try:
            # 1. Start container with resource limits
            container = self.client.containers.run(
                image=benchmarks[benchmark]["docker_image"],
                command=f"sleep {test.estimated_duration}",  # Simulated test execution
                cpu_period=100000,
                cpu_quota=int(resource.cpu_limit * 100000),
                mem_limit=f"{int(resource.cpu_limit * 1024)}M",
                detach=True
            )
            
            # 2. Monitor execution
            start_time = time.time()
            container.wait()  # Wait for completion
            actual_time = time.time() - start_time
            
            # 3. Estimate energy consumption
            # Energy (Joules) = Power (Watts) × Time (seconds)
            power = resource.power_model * resource.cpu_limit * test.cpu_intensity
            energy = power * actual_time
            
            # 4. Simulate coverage measurement (in real implementation, use JaCoCo)
            # Coverage is inversely related to execution speed but with noise
            coverage = min(1.0, 0.8 + 0.2 * random.random() - (actual_time / 100))
            
            return {
                "time": actual_time,
                "energy": energy,
                "coverage": max(0.0, coverage)
            }
            
        finally:
            if container:
                container.remove(force=True)

# ==================== 4. E-NSGA-II Core Algorithm ====================
class ENSGA2:
    def __init__(self, test_cases: List[TestCase], resources: List[DockerResource], 
                 benchmark: str, pop_size=100, max_generations=50):
        self.test_cases = test_cases
        self.resources = resources
        self.benchmark = benchmark
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.runner = DockerTestRunner()
        
    def initialize_population(self) -> List[Chromosome]:
        """Create random initial population"""
        population = []
        for _ in range(self.pop_size):
            # Random selection (approx 50% of tests)
            selection = [random.random() > 0.5 for _ in self.test_cases]
            selected_indices = [i for i, sel in enumerate(selection) if sel]
            
            # Random ordering
            random.shuffle(selected_indices)
            
            # Random resource assignment
            resource_assignment = [random.choice(self.resources).id 
                                  for _ in selected_indices]
            
            population.append(Chromosome(selection, selected_indices, resource_assignment))
        return population
    
    def evaluate_chromosome(self, chrom: Chromosome) -> Tuple[float, float, float]:
        """
        Evaluate a single chromosome by executing tests in Docker
        Returns: (coverage, time, energy)
        """
        total_time = 0
        total_energy = 0
        total_coverage = 0
        
        if not chrom.ordering:  # No tests selected
            return 0.0, 0.0, 0.0
        
        # Execute tests in the specified order
        for test_idx, resource_id in zip(chrom.ordering, chrom.resource_assignment):
            test = self.test_cases[test_idx]
            resource = next(r for r in self.resources if r.id == resource_id)
            
            # Run test in Docker container
            result = self.runner.run_test_in_container(test, resource, self.benchmark)
            
            total_time += result["time"]
            total_energy += result["energy"]
            total_coverage += result["coverage"]
        
        # Average coverage across tests
        avg_coverage = total_coverage / len(chrom.ordering)
        
        return avg_coverage, total_time, total_energy
    
    def non_dominated_sort(self, population: List[Chromosome], 
                          fitnesses: List[Tuple]) -> List[List[int]]:
        """NSGA-II non-dominated sorting"""
        # Implementation of fast non-dominated sort
        fronts = [[]]
        domination_counts = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        
        # Compare all pairs
        for i in range(len(population)):
            for j in range(len(population)):
                if i == j:
                    continue
                
                if self.dominates(fitnesses[i], fitnesses[j]):
                    dominated_solutions[i].append(j)
                elif self.dominates(fitnesses[j], fitnesses[i]):
                    domination_counts[i] += 1
            
            if domination_counts[i] == 0:
                fronts[0].append(i)
        
        # Build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def dominates(self, a: Tuple, b: Tuple) -> bool:
        """Check if solution A dominates solution B (max coverage, min time, min energy)"""
        # a and b are (coverage, time, energy)
        better_or_equal = (a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2])
        strictly_better = (a[0] > b[0] or a[1] < b[1] or a[2] < b[2])
        return better_or_equal and strictly_better
    
    def crowding_distance(self, front: List[int], fitnesses: List[Tuple]) -> List[float]:
        """Calculate crowding distance for solutions in a front"""
        distances = [0.0] * len(front)
        num_objectives = 3
        
        for obj_idx in range(num_objectives):
            # Sort by current objective
            sorted_front = sorted(front, key=lambda i: fitnesses[i][obj_idx])
            
            # Boundary solutions get infinite distance
            distances[front.index(sorted_front[0])] = float('inf')
            distances[front.index(sorted_front[-1])] = float('inf')
            
            # Normalize objective values
            min_val = fitnesses[sorted_front[0]][obj_idx]
            max_val = fitnesses[sorted_front[-1]][obj_idx]
            norm = max_val - min_val
            
            if norm > 0:
                for i in range(1, len(sorted_front) - 1):
                    idx = front.index(sorted_front[i])
                    next_idx = front.index(sorted_front[i + 1])
                    prev_idx = front.index(sorted_front[i - 1])
                    
                    distances[idx] += (fitnesses[next_idx][obj_idx] - 
                                      fitnesses[prev_idx][obj_idx]) / norm
        
        return distances
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """Two-point crossover for selection, OX for ordering, uniform for resources"""
        # Crossover selection (two-point)
        length = len(parent1.selection)
        pt1, pt2 = sorted(random.sample(range(length), 2))
        
        child_selection = parent1.selection[:pt1] + parent2.selection[pt1:pt2] + \
                         parent1.selection[pt2:]
        
        # Ordered Crossover (OX) for test ordering
        child_ordering = self.ordered_crossover(parent1.ordering, parent2.ordering)
        
        # Uniform crossover for resource assignment
        child_resources = []
        min_len = min(len(parent1.resource_assignment), len(parent2.resource_assignment))
        for i in range(min_len):
            child_resources.append(parent1.resource_assignment[i] 
                                  if random.random() > 0.5 
                                  else parent2.resource_assignment[i])
        
        return Chromosome(child_selection, child_ordering, child_resources)
    
    def ordered_crossover(self, order1: List[int], order2: List[int]) -> List[int]:
        """Ordered Crossover (OX) for permutation"""
        if not order1 or not order2:
            return order1 or order2
        
        size = min(len(order1), len(order2))
        pt1, pt2 = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[pt1:pt2] = order1[pt1:pt2]
        
        current_pos = pt2
        for gene in order2:
            if gene not in child:
                if current_pos >= size:
                    current_pos = 0
                child[current_pos] = gene
                current_pos += 1
        
        return child
    
    def mutate(self, chrom: Chromosome, mutation_rate=0.1) -> Chromosome:
        """Apply mutation operators"""
        # Mutation on selection (bit flip)
        new_selection = chrom.selection.copy()
        for i in range(len(new_selection)):
            if random.random() < mutation_rate:
                new_selection[i] = not new_selection[i]
        
        # Mutation on ordering (swap)
        new_ordering = chrom.ordering.copy()
        if len(new_ordering) >= 2 and random.random() < mutation_rate:
            i, j = random.sample(range(len(new_ordering)), 2)
            new_ordering[i], new_ordering[j] = new_ordering[j], new_ordering[i]
        
        # Mutation on resource assignment (random resetting)
        new_resources = chrom.resource_assignment.copy()
        for i in range(len(new_resources)):
            if random.random() < mutation_rate:
                new_resources[i] = random.choice(self.resources).id
        
        return Chromosome(new_selection, new_ordering, new_resources)
    
    def run(self) -> List[Tuple[Chromosome, Tuple]]:
        """Main E-NSGA-II execution loop"""
        # Initialize population
        population = self.initialize_population()
        
        for generation in range(self.max_generations):
            print(f"Generation {generation + 1}/{self.max_generations}")
            
            # 1. Evaluate fitness
            fitnesses = [self.evaluate_chromosome(chrom) for chrom in population]
            
            # 2. Non-dominated sorting
            fronts = self.non_dominated_sort(population, fitnesses)
            
            # 3. Crowding distance calculation
            crowding_distances = []
            for front in fronts:
                distances = self.crowding_distance(front, fitnesses)
                crowding_distances.extend(distances)
            
            # 4. Selection (tournament selection)
            selected_indices = self.tournament_selection(population, fitnesses, 
                                                        crowding_distances)
            
            # 5. Create new population through crossover and mutation
            new_population = []
            for i in range(0, len(selected_indices), 2):
                if i + 1 < len(selected_indices):
                    parent1 = population[selected_indices[i]]
                    parent2 = population[selected_indices[i + 1]]
                    
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent2, parent1)
                    
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    new_population.extend([child1, child2])
            
            # 6. Elitism: Keep best solutions from previous generation
            elite_size = self.pop_size - len(new_population)
            if elite_size > 0:
                # Combine all solutions and sort by front and crowding distance
                all_solutions = list(zip(population, fitnesses, crowding_distances))
                all_solutions.sort(key=lambda x: (fronts.index(x[0]), -x[2]))
                elites = [sol[0] for sol in all_solutions[:elite_size]]
                new_population.extend(elites)
            
            population = new_population
        
        # Final evaluation
        final_fitnesses = [self.evaluate_chromosome(chrom) for chrom in population]
        fronts = self.non_dominated_sort(population, final_fitnesses)
        
        # Return Pareto front (first front)
        pareto_front = []
        for idx in fronts[0]:
            pareto_front.append((population[idx], final_fitnesses[idx]))
        
        return pareto_front
    
    def tournament_selection(self, population, fitnesses, crowding_distances, 
                            tournament_size=2) -> List[int]:
        """Binary tournament selection"""
        selected = []
        for _ in range(len(population)):
            # Randomly pick tournament participants
            participants = random.sample(range(len(population)), tournament_size)
            
            # Select winner based on front and crowding distance
            winner = min(participants, key=lambda i: (
                self.get_front_rank(i, fitnesses),  # Lower front is better
                -crowding_distances[i]  # Higher crowding distance is better
            ))
            
            selected.append(winner)
        
        return selected
    
    def get_front_rank(self, idx, fitnesses):
        """Helper to get front rank for a solution"""
        # Simplified implementation
        fronts = self.non_dominated_sort([Chromosome([], [], [])], fitnesses)
        for rank, front in enumerate(fronts):
            if idx in front:
                return rank
        return len(fronts)

# ==================== 5. Main Execution ====================
if __name__ == "__main__":
    print("Starting E-NSGA-II for Docker-based Test Optimization")
    print("=" * 60)
    
    # Run optimization for each benchmark
    for benchmark_name in benchmarks.keys():
        print(f"\nOptimizing for benchmark: {benchmark_name}")
        print("-" * 40)
        
        test_cases = benchmarks[benchmark_name]["test_cases"]
        optimizer = ENSGA2(test_cases, resources, benchmark_name, 
                          pop_size=30, max_generations=10)  # Small scale for demo
        
        pareto_front = optimizer.run()
        
        # Display results
        print(f"\nPareto Front Solutions for {benchmark_name}:")
        print("Rank | Coverage (%) | Time (s) | Energy (J) | #Tests")
        print("-" * 60)
        
        for rank, (chrom, (cov, t, e)) in enumerate(pareto_front[:5]):  # Top 5
            num_tests = len(chrom.ordering)
            print(f"{rank+1:4} | {cov*100:11.2f} | {t:8.2f} | {e:10.2f} | {num_tests:6}")
    
    print("\n" + "=" * 60)
    print("Optimization completed successfully!")
