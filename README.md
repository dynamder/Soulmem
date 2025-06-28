# Soulmem

This is a memory system for LLM, built for more humanized LLM response and behaviour.



It's now developing, the current project structure is showed below.



## Structure

### Long Term Memory

Soulmem use Qdrant for the vector db. soulmem organize all memories in a graph structure. This is the only persistent memory storage.

Over a period, we will decay the memories ( for link intensity and memory content), and decide if we should evolve some memories (not designed yet)



### Working Memory

working memory maintain a taskset, representing the current and recent tasks & event. All task has a focus score attached to it. Using Softmax to normalize them, the one with the highest probability become the focus.



When a user message comes, LLM first find whether this message is relevant to tasks in the working memory, and determine whether it should extract more memories from the long term memory.

If it choose to extract, then we extract some relevant memories from db, add them to the working memory, also as a graph.

Then we decide if we will shift the task focus. recent focus will have an inertia factor to simulate focus consistency over time. The current message will give a relevant boost score, and we normalize again to get new focus.



we sample relevant memories attached to the task proportionally of the focus score, then we dfs on the graph for certain depth, trying to get more linked memories. These memories will send to the LLM to generate a response.



During the conversation, new information will pushed to the temporary part in the working memory, we maintain a reference count of them, if they are mentioned or used, we increment the count. temporary memories that has a count over the threshold will be consolidated to the long term memory.



Over a period, we clean the working memory graph for invalid node and edge.