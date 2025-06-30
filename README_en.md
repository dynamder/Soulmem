# Soulmem - A Human-like Memory System for LLMs

A memory architecture designed for large language models (LLMs) to achieve more natural and coherent responses and behaviors by simulating human memory mechanisms.

## Core Concepts

### Memory Hierarchy
| Type                 | Storage               | Characteristics                                   | Lifecycle |
| -------------------- | --------------------- | ------------------------------------------------- | --------- |
| **Long-term Memory** | Vector Database       | Persistent storage, knowledge graph structure     | Permanent |
| **Working Memory**   | RAM                   | Currently activated memory subgraph               | Temporary |
| **Temporary Memory** | Within Working Memory | Transient information from real-time conversation | Temporary |

### Core Mechanisms
- **Task**  
  Represents an event unit currently being processed by the LLM, containing associated memories and attention weights
- **Task Focus**  
  The core task with the highest attention score
- **Activation**  
  The retrieval process from long-term memory → working memory, and working memory → LLM context
- **Consolidation**  
  The transformation process from working memory → long-term memory
- **Evolution**  
  The self-optimization process for long-term memory (executed periodically)

## Workflow

```mermaid
graph TD
    A[User Input] --> B{Task Analysis}
    B -->|Relevant tasks exist| C[Sort Tasks by Relevance]
    B -->|No relevant task| D[Create New Task]
    C --> E{Need long-term memory activation?}
    D --> E
    E -->|Yes| F[Extract Memories from Vector DB]
    E -->|No| G[Update Working Memory]
    F --> G
    G --> H[Calculate Task Attention Scores]
    H --> I[Determine Task Focus]
    I --> J[Extract Memories by Attention Weight]
    J --> K[Depth-limited DFS Traversal of Memory Graph]
    K --> L[Assemble Memory Context]
    L --> M[Generate LLM Response]
    
    subgraph Background Processes
        N[Memory Activation Event] --> O[Record Co-activated Memories]
        O --> P{Reached Time Threshold?}
        P -->|Yes| Q[Filter Memories for Consolidation via LLM]
        R{Evolution Cycle Reached} --> S[Execute Memory Evolution]
    end