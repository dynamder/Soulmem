use std::collections::HashMap;
use mockall::automock;
use crate::memory::NodeRefId;

#[allow(dead_code)]
#[automock]
///记录一份记忆的激活记录
pub trait NoteRecord {
    fn activation_count(&self) -> u32; //被激活的次数
    fn record_activation(&mut self, act_ids: &[NodeRefId]); //记录共激活情况
    fn activation_history(&self) -> &HashMap<NodeRefId, u32>; //共激活历史
}