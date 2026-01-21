use std::collections::VecDeque

//滑动窗口（容器、容量、标记计数、摘要用临时储存）
pub struct SlidingWindow<information> {
    window: VecDeque<information>,
    capacity: usize,
    tag_count: usize,
    summary: Vec<information>
}

impl<information> SlidingWindow {
    //新建
    pub fn new(capacity: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity: 0,
            tag_count: 0,
            summary: Vec::with_capacity(capacity)
        }
    }
    //信息滑入，若滑出时信息有标记则发送摘要用片段
    pub fn push(&mut self, value: information) -> Option<Vec<information>> {
        value = self.auto_tag(value);
        let is_tagged: bool = false;
        let target: Option<information> = None;
        if self.window.len() == self.capacity {
            is_tagged = self.pop();
        }
        if is_tagged {
            target = self.summarize();
        }
        self.capacity += 1;
        self.window.push_back(value);
        target
    }
    //信息滑出，返回弹出信息是否被标记
    pub fn pop(&mut self) -> bool {
        let target = self.window.pop_front();
        self.capacity -= 1;
        self.summary.push(target.clone());
        self.get_tagged(target)
    }
    //获取窗口大小
    pub fn len(&self) -> usize {
        self.window.len()
    }
    //获取窗口容量
    pub fn get_capacity(&self) -> usize {
        self.capacity
    }
    //获取窗口容量（可变）
    pub fn get_mut_capacity(&mut self) -> &mut usize {
        &mut self.capacity
    }
    //判断窗口是否为空
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
    //清空窗口
    pub fn clear(&mut self) {
        self.window.clear();
        self.capacity = 0;
        self.tag_count = 0;
        self.summary.clear();
    }
    //标记用
    pub fn tag_information(&mut self, index: usize) {
        if index < self.capacity {
            self.window[index].tag_information();
        }
    }
    //取消标记用
    pub fn untag_information(&mut self, index: usize) {
        if index < self.capacity {
            self.window[index].untag_information();
        }
    }
    //每滑出capacity次信息时进行一次标记
    fn auto_tag(&mut self, value: information) -> information {
        self.tag_count += 1;
        if self.tag_count == self.capacity {
            value.tag_information();
            self.tag_count = 0;
        }
        value
    }
    //给出摘要
    pub fn summarize(&mut self) -> Vec<information> {
        self.summary.drain(..).collect()
    }
    //检测是否存在标记信息
    pub fn is_tagged(&mut self, value: information) -> bool {
        value.is_tagged()
    }

}

pub struct information {
    pub text: String,
    pub tag: bool,
}

impl information {
    pub fn new(text: String) -> Self {
        Self { text, false }
    }
    pub fn tag_information(&mut self) {
        self.tag = true;
    }
    pub fn untag_information(&mut self) {
        self.tag = false;
    }
    pub fn is_tagged(&self) -> bool {
        self.tag
    }
    pub fn get_mut_capacity(&mut self) -> &mut usize {
        &mut self.capacity
    }
}
