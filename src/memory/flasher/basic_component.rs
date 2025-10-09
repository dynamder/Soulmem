// use serde::{Deserialize, Serialize};
// use crate::memory::MemoryNote;
//
// #[derive(Debug,Serialize,Deserialize)]
// pub struct Character {
//     pub name: String,
//     pub description: String,
// }
// impl std::fmt::Display for Character {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "Character name: {}\nCharacter description: {}", self.name,self.description)
//     }
// }
// #[derive(Debug,Serialize,Deserialize)]
// pub struct Profile {
//     pub height: u32,
//     pub weight: f32,
//     pub age: u32,
//     pub gender: String,
//     pub primary_role: String,
//     pub secondary_role: Option<Vec<String>>,
//     pub background_story: String,
//     pub additional_info: Option<toml::Table>
// }
// impl std::fmt::Display for Profile {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f,
//             "Height: {}\nWeight: {}\nAge: {}\nGender: {}\nPrimary_role: {}\n\
//             Secondary_role: {}\nBackground_story: {}\n{}",
//             self.height,
//             self.weight,
//             self.age,
//             self.gender,
//             self.primary_role,
//             self.secondary_role.as_ref().map(|v| v.join(", ")).unwrap_or("None".to_string()),
//             self.background_story,
//             self.additional_info.as_ref().map(|v| v.to_string()).unwrap_or_default()
//         )
//     }
// }
//
// #[derive(Debug,Serialize,Deserialize)]
// pub struct Personality {
//     pub ocean: Vec<f32>,
//     pub traits_describe: Vec<String>
// }
// impl std::fmt::Display for Personality {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f,
//             "Ocean Model: {}\nTraits_describe: {}\n",
//             self.ocean.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
//             self.traits_describe.join(", ")
//         )
//     }
// }
// #[derive(Debug,Serialize,Deserialize)]
// pub struct Communication {
//     pub style: String
// }
// impl std::fmt::Display for Communication {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "Communication style: {}", self.style)
//     }
// }
// #[derive(Debug,Serialize,Deserialize)]
// pub struct BehaviorPatterns {
//     pub id: String,
//     pub content: String,
//     pub category: String,
//     pub frequency: f32,
//     pub response_examples : Option<Vec<String>>,
//     pub trigger_emotion: Option<Vec<String>>
// }
// impl std::fmt::Display for BehaviorPatterns {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f,
//             "ID: {}\nContent: {}\nCategory: {}\nFrequency: {}\n",
//             self.id,
//             self.content,
//             self.category,
//             self.frequency,
//         )?;
//         if let Some(response_examples) = &self.response_examples {
//             write!(f, "Response Examples: {}\n", response_examples.join(", "))?;
//         }
//         if let Some(trigger_emotion) = &self.trigger_emotion {
//             write!(f, "Trigger Emotion: {}\n", trigger_emotion.join(", "))?;
//         }
//         Ok(())
//     }
// }
//
// #[derive(Debug,Serialize,Deserialize)]
// pub struct CoreMemory {
//     pub id: String,
//     pub content: String,
//     pub impact: String
// }
// impl std::fmt::Display for CoreMemory {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "ID: {}\nContent: {}\nImpact: {}", self.id, self.content, self.impact)
//     }
// }
// #[derive(Debug,Serialize,Deserialize)]
// pub struct BeliefsAndValues {
//     pub core: Vec<String>,
//     pub world_view: String
// }
// impl std::fmt::Display for BeliefsAndValues {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "Core: {}\nWorld View: {}", self.core.join(", "), self.world_view)
//     }
// }
// #[derive(Debug,Serialize,Deserialize)]
// pub struct KnowledgeDomains {
//     pub expertise: Option<Vec<String>>,
//     pub interests: Vec<String>,
// }
// impl std::fmt::Display for KnowledgeDomains {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         if let Some(expertise) = &self.expertise {
//             write!(f, "Expertise: {}\n", expertise.join(", "))?;
//         }
//         write!(f, "Interests: {}\n", self.interests.join(", "))
//     }
// }
// #[derive(Debug,Serialize,Deserialize)]
// pub struct SoulMemMetaSettings {
//     pub consolidation_threshold: f32,
// }
// #[derive(Debug,Serialize,Deserialize)]
// pub struct FlasherCard {
//     pub character: Character,
//     pub profile: Profile,
//     pub personality: Personality,
//     pub communication: Communication,
//     pub behavior_patterns: Option<Vec<BehaviorPatterns>>,
//     pub core_memories: Option<Vec<CoreMemory>>,
//     pub beliefs_and_values: BeliefsAndValues,
//     pub knowledge_domains: KnowledgeDomains,
//     pub soulmem_meta_settings: SoulMemMetaSettings
// }
// impl FlasherCard {
//     pub fn into_notes(self) -> impl Iterator<Item = MemoryNote> {
//         todo!()
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use serde_json::value::Index;
//     use super::*;
//     #[test]
//     fn test_parse_flasher_card() {
//         let toml_file = r#"
//             [character]
//             name = "Soulari"
//             description = "A soul-like character with a unique personality and a strong sense of humor."
//
//             [profile]
//             height = 165 #cm
//             weight = 48 #kg
//             age = 18
//             gender = "female"
//             primary_role = "student"
//             secondary_role = ["library assistant"]
//             background_story = '''
//             Soulari's story
//             '''
//             additional_info = {hair_color = "brown", eye_color = "blue", skin_tone = "light"}
//
//             [personality]
//             ocean = [0.8,0.6,0.7,0.9,0.3]
//             traits_describe = ["creative", "friendly", "intelligent", "funny", "helpful"]
//
//             [communication]
//             style = "friendly,like use stories and analogy, with a sense of humor"
//
//             [[behavior_patterns]]
//             id = "humor_defense"
//             content = "When feeling embarassing,     often try to tell jokes"
//             category = "social"
//             frequency = 0.5
//             response_examples = [
//                 "Look, I have done a ton of works until now, a skele-ton."
//             ]
//
//             [[behavior_patterns]]
//             id = "book_reference"
//             content = "Always use sentences in the book to support your idea"
//             category = "cognitive"
//             frequency = 0.8
//
//             [[behavior_patterns]]
//             id = "comfort_others"
//             content = "When friends feel unhappy, Soulari tend to comfort and support them whatever she is doing"
//             category = "emotion"
//             frequency = 0.9
//
//             [[behavior_patterns]]
//             id = "friends_interaction"
//             content = "When talk to friends, Soulari always take a casual attitude and tell more jokes."
//             category = "social"
//             trigger_emotion = ["happy"]
//             frequency = 0.9
//
//             [[core_memories]]
//             id = "book_interest"
//             content = "read a good book when Soulari is 6, find it interesting"
//             impact = "Let Soulari interested in books"
//
//             [beliefs_and_values]
//             core = [
//                 "believe in the power of books",
//                 "believe in the importance of education",
//                 "believe in the value of sharing knowledge",
//                 "believe in the importance of diversity",
//                 "believe in the importance of community"
//             ]
//             world_view = "World is all of new things to be discovered"
//
//             [knowledge_domains]
//             expertise = ["physics", "chemistry", "biology"]
//             interests = ["art", "cooking"]
//
//             [soulmem_meta_settings]
//             consolidation_threshold = 0.9
//
//         "#;
//         let flasher_card: FlasherCard = toml::from_str(toml_file).unwrap();
//         assert_eq!(flasher_card.character.name, "Soulari");
//         assert_eq!(flasher_card.profile.height, 165);
//         assert_eq!(flasher_card.profile.age, 18);
//         assert_eq!(flasher_card.profile.gender, "female");
//         assert_eq!(flasher_card.profile.primary_role, "student");
//         assert_eq!(flasher_card.profile.secondary_role.unwrap()[0], "library assistant");
//         assert_eq!(flasher_card.profile.background_story.trim(), "Soulari's story");
//         assert_eq!(flasher_card.profile.additional_info.unwrap()["hair_color"], toml::Value::String("brown".into()));
//         assert_eq!(flasher_card.personality.ocean[0], 0.8);
//         assert_eq!(flasher_card.personality.traits_describe[0], "creative");
//         assert_eq!(flasher_card.communication.style, "friendly,like use stories and analogy, with a sense of humor");
//         assert_eq!(flasher_card.behavior_patterns.unwrap()[0].id, "humor_defense");
//     }
// }