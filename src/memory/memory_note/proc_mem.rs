
//动作类型
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub enum ActionType {
    Speak, //语气类，说话方式
    Skill(SkillRecord), //技能类，例如使用外部工具
    Think, //思维类，复杂任务中的思考方式倾向等
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
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct SkillRecord {
    //TODO: 后续版本功能，仅做PlaceHolder
}

///程序性记忆的动作节点(Action)
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
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

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct ProcMemory {
    action: Action
}
impl ProcMemory {
    pub fn new(action: Action) -> Self {
        Self {
            action,
        }
    }
}
impl From<Action> for ProcMemory {
    fn from(action: Action) -> Self {
        Self::new(action)
    }
}