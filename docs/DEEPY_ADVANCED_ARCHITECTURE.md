# Deepy Advanced Fine-Tuning Architecture

This document captures the long-term redesign plan for Deepy as an adaptive, parameter-aware, video-intelligent generation agent inside WanGP.

## Goals

Deepy should evolve from a mostly template-driven assistant into a local-first media orchestration system that:

- applies every supported parameter intentionally
- refuses to silently ignore unsupported parameters
- routes requests across Wan, LTX, CogVideo, Hunyuan, Mochi, and future backends with capability checks
- understands video as temporal media rather than isolated frames
- preserves workflow continuity through memory
- supports reliable headless and queued automation
- optimizes runtime decisions for the active GPU, model, resolution, sequence length, and requested quality
- uses remote AI only as an optional reasoning layer, not as the realtime orchestration core

## Target Pipeline

```text
User Prompt / API Input
        ↓
Intent + Parameter Parser
        ↓
Dynamic Routing Engine
        ↓
Memory & Context System
        ↓
Video Understanding Layer
        ↓
Optimization Layer
        ↓
Execution Manager
        ↓
Generation Backend
```

## Parameter Policy

The target merge order is:

```text
prompt parameters
  > automation parameters
  > selected profile
  > UI settings
  > defaults
```

Deepy must validate and explain every requested parameter. If a requested setting is not available for the selected tool/backend, Deepy should not generate silently with template defaults. Instead it should:

1. identify the unsupported setting
2. explain why it cannot be applied to the active backend/tool
3. suggest an exposed alternative when one exists
4. translate the request to an equivalent backend setting only when the active template exposes that setting
5. ask the user to edit the template directly or explicitly continue without the unavailable setting

## Initial Backend Capability Layer

Deepy now includes a conservative backend capability mapper for selected generation tools. The mapper identifies broad backend families from the active tool's model definition/template metadata and reports capability hints such as skip-step support, sliding-window support, temporal upsampling support, audio prompt support, and the most likely motion-control parameter.

The assistant can query this through `Get Backend Capabilities` before applying backend-specific settings. This does not replace `Get Default Settings`; it is a routing guard. `Get Default Settings` remains the source of truth for exact exposed `extra_settings` keys.

Example use:

```text
User asks: "make this Wan video faster with skip-step caching"
Deepy:
  1. checks backend capabilities for gen_video
  2. confirms the active backend is Wan-family and skip-step capable
  3. checks Get Default Settings for exposed skip-step labels
  4. applies the exposed setting or blocks with a clear explanation
```

## Video Understanding Roadmap

The target video understanding pipeline is:

```text
Video
   ↓
Frame Sampling
   ↓
Temporal Encoder
   ↓
Motion Analysis
   ↓
Scene Understanding
   ↓
Semantic Embedding
   ↓
Memory Storage
```

The analysis layer should reason about:

- temporal consistency
- motion flow
- camera movement
- object and character persistence
- scene transitions
- emotional pacing
- lighting continuity
- motion quality
- prompt mismatch during edits

Local models such as Qwen VL, InternVL, MiniCPM-V, VideoLLaMA, and VideoChat2 are preferred for realtime inspection and orchestration. Remote models may be used optionally for complex cinematic planning and semantic decomposition.

## Memory Roadmap

Deepy should persist workflow memory for:

- character descriptions and embeddings
- scene layouts
- camera styles
- motion profiles
- lighting patterns
- previous clips
- failed generations
- prompt and edit history
- user preferences

A future memory implementation should store semantic, style, motion, and temporal embeddings in a local vector store such as FAISS or ChromaDB.

## Headless Automation Roadmap

Headless Deepy should support:

- API/CLI input
- batch generation
- queue handling
- retry logic
- GPU-aware scheduling
- automatic recovery
- background execution
- webhook notifications
- distributed execution hooks

Target flow:

```text
API / CLI Input
        ↓
Deepy Planner
        ↓
Task Queue
        ↓
Execution Scheduler
        ↓
Monitoring Layer
        ↓
Result Packaging
```

## Runtime Optimization Roadmap

Deepy should eventually select runtime strategies dynamically based on model, GPU, VRAM, resolution, sequence length, and desired quality:

- Flash Attention
- Sage Attention
- xFormers
- SDPA
- Triton
- CUDA Graphs
- vLLM for repeated local LLM calls
- adaptive sampling steps
- adaptive skip steps
- denoise schedule adjustments
- temporal precision and interpolation adjustments

## Hybrid Local + Remote Principle

Local systems should own:

- orchestration
- parameter routing
- runtime optimization
- queue management
- generation scheduling
- editing execution
- backend selection

Remote APIs should remain optional and limited to:

- creative planning
- cinematic reasoning
- story generation
- complex scene decomposition
- advanced semantic analysis
- difficult editing strategy

Remote APIs should not control realtime orchestration, generation loops, backend scheduling, or continuous task handling because that increases cost, latency, dependency, privacy risk, and failure points.
