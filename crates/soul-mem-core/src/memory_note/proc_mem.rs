use serde::{Deserialize, Serialize};

//动作类型
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Speak,              //语气类，说话方式
    Skill(SkillRecord), //技能类，例如使用外部工具
    Think,              //思维类，复杂任务中的思考方式倾向等
}
impl ActionType {
    pub fn new_speak() -> Self {
        Self::Speak
    }
    pub fn new_skill(skill_record: SkillRecord) -> Self {
        Self::Skill(skill_record)
    }
    pub fn new_think() -> Self {
        Self::Think
    }
}
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Serialize, Deserialize)]
pub struct SkillRecord {
    //TODO: 后续版本功能，仅做PlaceHolder
}

///程序性记忆的动作节点(Action)
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Serialize, serde::Deserialize)]
pub struct Action {
    content: String,
    action_type: ActionType,
}
impl Action {
    pub fn new(content: String, action_type: ActionType) -> Self {
        Self {
            content,
            action_type,
        }
    }
    pub fn get_content(&self) -> &str {
        &self.content
    }
    pub fn get_action_type(&self) -> &ActionType {
        &self.action_type
    }
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Serialize, Deserialize)]
pub struct ProcMemory {
    action: Action,
}
impl ProcMemory {
    pub fn new(action: Action) -> Self {
        Self { action }
    }
}
impl From<Action> for ProcMemory {
    fn from(action: Action) -> Self {
        Self::new(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_get_content() {
        let action = Action::new("speak softly".to_string(), ActionType::new_speak());
        assert_eq!(action.get_content(), "speak softly");
        assert_eq!(action.get_action_type(), &ActionType::Speak);
    }

    #[test]
    fn test_action_new_skill_and_think() {
        let skill = Action::new("use_tool".to_string(), ActionType::new_skill(SkillRecord {}));
        assert_eq!(skill.get_content(), "use_tool");
        assert_eq!(skill.get_action_type(), &ActionType::Skill(SkillRecord {}));

        let think = Action::new("plan".to_string(), ActionType::new_think());
        assert_eq!(think.get_action_type(), &ActionType::Think);
    }

    #[test]
    fn test_proc_memory_from_action() {
        let action = Action::new("act".to_string(), ActionType::new_speak());
        let mem: ProcMemory = action.into();
        let _ = mem;
    }
}
