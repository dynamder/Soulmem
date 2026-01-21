use std::collections::VecDeque

//滑动窗口（容器、容量、标记窗口）通过标记窗口中对应索引是否为true来判断是否被标记
pub struct SlidingWindow<T> {
    window: VecDeque<T>,
    capacity: usize,
    tag_window: VecDeque<bool>,
    tag_count: usize
}

impl<T> SlidingWindow {
    //新建
    pub fn new(capacity: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
            tag_window: VecDeque::with_capacity(capacity),
            tag_count: 0
        }
    }
    //信息滑入，对应标记窗口滑入false，返回被弹出的信息，若未满或弹出未标记的信息则返回None，若弹出被标记信息则返回Some(该信息)
    pub fn push(&mut self, value: T) -> Option<T> {
        let target = None;
        if self.window.len() == self.capacity {
            target = self.pop();
        }
        self.capacity += 1;
        self.window.push_back(value);
        self.auto_tag();
        target
    }
    //信息滑出，返回被弹出的信息，若未满或弹出未标记的信息则返回None，若弹出被标记信息则返回Some(该信息)
    pub fn pop(&mut self) -> Option<T> {
        if self.tag_window.get(0).copied() == Some(true) {
            let target = Some(self.dwindow.pop_front());
            self.capacity -= 1;
            self.tag_window.pop_front();
            target
        } else {
            self.window.pop_front();
            self.capacity -= 1;
            self.tag_window.pop_front();
            None
        }
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
    //标记用
    pub fn tag_information(&mut self, index: usize) {
        if index < self.capacity {
            self.tag_window[index] = true;
        }
    }
    //取消标记用
    pub fn untag_information(&mut self, index: usize) {
        if index < self.capacity {
            self.tag_window[index] = false;
        }
    }
    //每滑出capacity次信息时进行一次标记
    fn auto_tag(&mut self) {
        self.tag_count += 1;
        if self.tag_count == self.capacity {
            self.tag_window.push_back(true);
            self.tag_count = 0;
        }
        else{
            self.tag_window.push_back(false);
        }
    }
}
