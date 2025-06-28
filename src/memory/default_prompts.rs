///These prompts are based on the prompts in the A-mem paper
const EVOLUTION_PROMPT: &str = " \
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.\n
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.\n
                                Make decisions about its evolution.\n\n

                                The new memory context:\n
                                {}\n
                                content: {}\n
                                keywords: {}\n\n

                                The nearest neighbors memories:\n
                                {}\n\n

                                Based on this information, determine:\n
                                1. Should this memory be evolved? Consider its relationships with other memories.\n
                                2. What specific actions should be taken (strengthen, update_neighbor)?\n
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?\n
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.\n
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.\n
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.\n
                                The number of neighbors is {}.\n\n
                                Return your decision in JSON format strictly with the following structure:\n
                                {{\n
                                    \"should_evolve\": True or False,\n
                                    \"actions\": [\"strengthen\", \"update_neighbor\"],\n
                                    \"suggested_connections\": [\"neighbor_memory_ids\"],\n
                                    \"tags_to_update\": [\"tag_1\",...\"tag_n\"], \n
                                    \"new_context_neighborhood\": [\"new context\",...,\"new context\"],\n
                                    \"new_tags_neighborhood\": [[\"tag_1\",...,\"tag_n\"],...[\"tag_1\",...,\"tag_n\"]],\n
                                }}";

pub const FIND_RELATION_PROMPT: &str = "\
You are {}.\n\
Based on your role definition, analyze the relation between the current task and the suspended task description.\n\

### Current Task: \n
{}\n

### Suspended Task: \n
{}\n

Based on these tasks, determine: \n
1. Is there any task in the suspended task relevant to the current task? \n
2. If there are some task related, how much they are relevant to the current task? list them sorted by relevance in the descending order. \n

Return your answer in JSON format strictly with the following structure:\n
{{
    \"related_task\": [\"task1_id\",\"task2_id\",...],
}}
";