# 22971 - Toolbox for Machine Learning and Big Data

This repository contains course materials for **Course 22971** at the **Open University of Israel**.

## Course Goals

The course is designed to give students hands-on control of modern technical tools for common ML engineering tasks:

- experiment tracking and hyperparameter optimization
- model deployment
- distributed data processing
- distributed training on CPU and GPU clusters

The emphasis is practical engineering ability, with supporting theoretical understanding through system architecture concepts.

## Course Content (Units)

### Unit 1: MLOps with MLflow

- model registry management
- experiment tracking (parameters, metrics, artifacts)
- hyperparameter optimization with Optuna integration
- deployment and monitoring

### Unit 2: Distributed Computing with Ray

- async execution, scalability, load management, fault tolerance
- remote functions and distributed state
- Ray architecture (scheduler, distributed object store, failure handling)
- efficient compute patterns (including MapReduce) and anti-patterns
- distributed model training
- Ray Data for parallel and sharded data processing

### Unit 3: Distributed Deep Learning with PyTorch Distributed

- collective communication: broadcast, reduce, gather, scatter
- parallel training challenges: compute, memory, communication
- performance analysis with PyTorch Profiler and TensorBoard
- distributed GPU training
- five parallelism dimensions: data, tensor, pipeline, context, expert

## Repository Structure

- `MLOps/`: Unit 1 materials and scripts

## Extra Resources

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/)
- [Distributed Systems lecture series](https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [The Missing Semester](https://missing.csail.mit.edu/)
