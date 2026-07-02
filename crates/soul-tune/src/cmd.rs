use std::{collections::HashMap, sync::Arc};

use crate::{
    base::SoulTuneEvent,
    utils::fuzzy::{FuzzyPatternBuilder, fuzzy_match},
};

pub struct UserCmd {
    name: String,
    aliases: Vec<String>,
    description: String,
    usage: String,
    args_completer: Option<Box<dyn Fn(&str) -> Vec<String>>>,
    handler: Box<dyn Fn(&[String]) -> Option<SoulTuneEvent>>,
}

impl UserCmd {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn aliases(&self) -> &[String] {
        &self.aliases
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn usage(&self) -> &str {
        &self.usage
    }

    pub fn args_completer(&self) -> Option<&dyn Fn(&str) -> Vec<String>> {
        self.args_completer.as_deref()
    }

    pub fn handler(&self) -> &dyn Fn(&[String]) -> Option<SoulTuneEvent> {
        &self.handler
    }
}

pub struct UserCmdBuilder {
    name: String,
    aliases: Vec<String>,
    description: String,
    usage: String,
    args_completer: Option<Box<dyn Fn(&str) -> Vec<String>>>,
    handler: Box<dyn Fn(&[String]) -> Option<SoulTuneEvent>>,
}

impl UserCmdBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            aliases: Vec::new(),
            description: String::new(),
            usage: String::new(),
            args_completer: None,
            handler: Box::new(|_| None),
        }
    }

    pub fn aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases = aliases.into_iter().map(|a| a.into()).collect();
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn usage(mut self, usage: impl Into<String>) -> Self {
        self.usage = usage.into();
        self
    }

    pub fn args_completer(
        mut self,
        args_completer: impl Fn(&str) -> Vec<String> + 'static,
    ) -> Self {
        self.args_completer = Some(Box::new(args_completer));
        self
    }

    pub fn handler(
        mut self,
        handler: impl Fn(&[String]) -> Option<SoulTuneEvent> + 'static,
    ) -> Self {
        self.handler = Box::new(handler);
        self
    }

    pub fn build(self) -> UserCmd {
        UserCmd {
            name: self.name,
            aliases: self.aliases,
            description: self.description,
            usage: self.usage,
            args_completer: self.args_completer,
            handler: self.handler,
        }
    }
}

pub struct CmdRegistry {
    commands: HashMap<String, UserCmd>,
}
impl CmdRegistry {
    pub fn new() -> Self {
        Self {
            commands: HashMap::new(),
        }
    }

    pub fn register(&mut self, cmd: UserCmd) {
        self.commands.insert(cmd.name.clone(), cmd);
    }

    pub fn get(&self, name: &str) -> Option<&UserCmd> {
        self.commands.get(name)
    }

    pub fn get_all(&self) -> &HashMap<String, UserCmd> {
        &self.commands
    }

    pub fn fuzzy_cmd_find(&self, query: &str) -> Vec<&UserCmd> {
        let match_result = fuzzy_match(
            FuzzyPatternBuilder::default().build(query),
            self.commands.keys(),
            false,
        );

        match_result
            .into_iter()
            .map(|(cmd, _)| self.commands.get(cmd).unwrap()) //SAFETY: fuzzy_match ensures cmd is in self.commands
            .collect()
    }
    pub fn fuzzy_cmd_completions(&self, query: &str) -> Vec<String> {
        todo!()
    }
}
