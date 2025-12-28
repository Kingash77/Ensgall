# E-NSGA-II: Energy-Aware Multi-Objective Test Optimization in Docker-Based Environments

This repository provides a **research-oriented, Docker-based simulation** of an
**Enhanced Non-dominated Sorting Genetic Algorithm II (E-NSGA-II)** for optimizing
software test execution under resource constraints.

The proposed approach simultaneously optimizes:
- **Test coverage maximization**
- **Execution time minimization**
- **Energy consumption minimization**

The implementation targets **containerized continuous integration (CI) and cloud testing environments**.

---

## 1. Research Motivation

Modern software testing pipelines increasingly rely on **containerized infrastructures**
(e.g., Docker-based CI/CD). While parallel test execution improves performance, it also
introduces **energy inefficiency and resource contention**.

This project explores the following research question:

> *How can test selection, ordering, and resource allocation be jointly optimized to
achieve high coverage while minimizing execution time and energy consumption?*

To address this, we model test execution as a **multi-objective optimization problem**
and solve it using an enhanced variant of **NSGA-II**.

---

## 2. Problem Formulation

Each solution (chromosome) encodes:
- **Test selection** (binary decision per test case)
- **Test execution order** (permutation of selected tests)
- **Docker resource assignment** (CPU-limited container profiles)

### Objectives:
Let:
- \( C \) = average test coverage  
- \( T \) = total execution time  
- \( E \) = total energy consumption  

The optimization problem is defined as:

- Maximize: \( C \)
- Minimize: \( T \)
- Minimize: \( E \)

---

## 3. Algorithm Overview

The optimization engine is based on **E-NSGA-II**, featuring:
- Fast non-dominated sorting
- Crowding distance preservation
- Elitist selection strategy
- Customized genetic operators for:
  - Test selection
  - Ordering (Ordered Crossover)
  - Resource assignment

Each chromosome is evaluated by executing tests inside **Docker containers**
with controlled CPU limits.

---

## 4. Docker-Based Execution Model

Each test case is executed in an isolated Docker container with:
- CPU quota enforcement
- Memory constraints
- Simulated execution time
- Energy estimation based on:
  
\[
Energy = Power \times ExecutionTime
\]

Energy consumption is modeled using **resource-specific power profiles**.

---

## 5. Benchmarks

The framework simulates industrial-scale systems inspired by:
- **Apache Commons Math**
- **JFreeChart**

Test execution characteristics (duration, CPU intensity) are probabilistically modeled
to enable controlled experimentation.

> ⚠️ Note: This implementation focuses on **conceptual validation** rather than
full-scale industrial measurement.

---

## 6. Experimental Workflow

1. Generate initial population
2. Evaluate chromosomes via Docker-based execution
3. Perform non-dominated sorting
4. Apply selection, crossover, and mutation
5. Iterate across generations
6. Extract Pareto-optimal solutions

The final output is a **Pareto front** representing optimal trade-offs
between coverage, time, and energy.

---

## 7. Running the Experiment

### Docker (Recommended)
```bash
docker pull YOUR_DOCKERHUB_USERNAME/e-nsga2-docker
docker run --rm YOUR_DOCKERHUB_USERNAME/e-nsga2-docker
