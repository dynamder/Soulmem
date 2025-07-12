
// 定义状态枚举
enum PipeIfState<I, R> {
    Original(I),
    Transformed(R),
}

pub struct PipeIf<I, F> {
    state: PipeIfState<I, F>, // 存储实际迭代器
}

/// 扩展 Iterator，条件执行的迭代器操作
pub trait IteratorPipe: Iterator + Sized {
    fn pipe_if<F, R>(self, condition: bool, f: F) -> PipeIf<Self, R>
    where
        F: FnOnce(Self) -> R,
        R: Iterator<Item = Self::Item>;
}

// 为所有迭代器实现扩展
impl<I: Iterator> IteratorPipe for I {
    fn pipe_if<F, R>(self, condition: bool, f: F) -> PipeIf<Self, R>
    where
        F: FnOnce(Self) -> R,
        R: Iterator<Item = Self::Item>,
    {
        if condition {
            let transformed = f(self);
            PipeIf {
                state: PipeIfState::Transformed(transformed),
            }
        } else {
            PipeIf {
                state: PipeIfState::Original(self),
            }
        }
    }
}

impl<I, R> Iterator for PipeIf<I, R>
where
    I: Iterator,
    R: Iterator<Item = I::Item>,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.state {
            PipeIfState::Original(iter) => iter.next(),
            PipeIfState::Transformed(iter) => iter.next(),
        }
    }
}
