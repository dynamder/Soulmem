///These prompts are based on the prompts in the A-mem paper
#[allow(unused)]
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

#[allow(unused)]
pub const FIND_RELATION_PROMPT: &str = "\
You are {}.\n\
Based on your role definition, analyze the relation between the current query and the suspended task description.\n\
### Current Query: \n
{}\n

### Suspended Task: \n
{}\n

Based on these tasks, determine: \n
1. Is there any task in the suspended task relevant to the current task? \n
2. If there are some task related, how much they are relevant to the current task? list them sorted by relevance in the descending order. \n
3. Extract keywords of the current query, no more than 5 words.

Return your answer in JSON format strictly with the following structure, with no additional character:\n
{{
    \"related_task\": [\"task1_id\",\"task2_id\",...],
    \"current_keywords\": [\"keyword1\",\",keyword2\",...]
}}
";

#[allow(unused)]
pub const RECONSOLIDATION_PROMPT: &str = " \
                                You are an AI memory evolution agent responsible for managing and reconsolidating a knowledge base.\n
                                Analyze the the new memory note according to keywords and context, also with the several memories that frequently co-activated.\n
                                Make decisions about its reconsolidation.\n\n

                                The new memory:\n\
                                {}

                                The frequently co-activated memories:\n
                                {}\n\n

                                Based on this information, determine:\n
                                1. Should this memory be modified? Consider its relationships with other memories.\n
                                2. What specific actions should be taken (strengthen_connection, update_self, both the two above)?\n
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the possible relationship between the two memories?\n\
                                   2.2 If choose to update the note itself, What does the new content, keywords, tags, context, base_emotion should be?

                                Content is the raw string of a memory note.\n
                                Keywords should be determined by important of frequent appeared concepts of the content, which can be used to analyze them later and categorize them.\n
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.\n
                                Context is the description of time and space circumstances of the memory note.\n
                                Base Emotion is the mood of the memory note, which can be used to retrieve them later with specific emotion.\n\n

                                If you choose not to update a specific field, don't let the field appear in the final JSON. But the \"id\" and \"should_modify\" are always required.\n

                                Return your decision in JSON format strictly with the following structure:\n
                                {{\n\
                                    \"id\": The id of the note itself,\n
                                    \"should_modify\": True or False,\n    
                                    \"action\": \"strengthen_connection\" or \"update_self\" or \"both\",\n
                                    \"suggested_connections\": [\"{{\"id\": \"memory_id\", \"relationship\": \"possible_relationship\", \"linked_category\": \"the category of the new neighbor\"}}\",...,\"{{\"id\": \"memory_id\", \"relationship\": \"possible_relationship\",\"linked_category\": \"the category of the new neighbor\"}}\"],\n
                                    \"new_content\":\"new_content\",
                                    \"new_keywords\":[\"keyword_1\",...,\"keyword_n\"],\n
                                    \"new_tags\": [\"tag_1\",...,\"tag_n\"], \n
                                    \"new_context\": \"new context\",\n\
                                    \"new_base_emotion\": \"new_base_emotion\"\n
                                }}";