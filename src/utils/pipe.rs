
enum PipeIfState<I, F, R> {
    Original(I),
    LazyTransform {
        iter: Option<I>,
        func: Option<F>
    },
    Transformed(R),
}

pub struct PipeIf<I, F, R> {
    state: PipeIfState<I, F, R>, // 存储实际迭代器
}

/// 扩展 Iterator，条件执行的迭代器操作
pub trait IteratorPipe: Iterator + Sized {
    fn pipe_if<F, R>(self, condition: bool, f: F) -> PipeIf<Self, F, R>
    where
        F: FnOnce(Self) -> R,
        R: Iterator<Item = Self::Item>;
    fn pipe<F, R>(self, f: F) -> PipeIf<Self, F, R>
    where
        F: FnOnce(Self) -> R,
        R: Iterator<Item = Self::Item>
    {
        self.pipe_if(true, f)
    }
}

// 为所有迭代器实现扩展
impl<I: Iterator> IteratorPipe for I {
    fn pipe_if<F, R>(self, condition: bool, f: F) -> PipeIf<Self, F, R>
    where
        F: FnOnce(Self) -> R,
        R: Iterator<Item = Self::Item>,
    {
        if condition {
            PipeIf {
                state: PipeIfState::LazyTransform {
                    iter: Some(self),
                    func: Some(f),
                }
            }
        } else {
            PipeIf {
                state: PipeIfState::Original(self),
            }
        }
    }
}

impl<I, F, R> Iterator for PipeIf<I, F, R>
where
    I: Iterator,
    F: FnOnce(I) -> R,
    R: Iterator<Item = I::Item>,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.state {
                PipeIfState::Original(iter) => return iter.next(),
                PipeIfState::Transformed(iter) => return iter.next(),
                PipeIfState::LazyTransform { iter, func } => {
                    let inner_iter = iter.take().expect("Missing iterator in LazyTransform");
                    let inner_func = func.take().expect("Missing function in LazyTransform");

                    let transformed_iter = inner_func(inner_iter);
                    self.state = PipeIfState::Transformed(transformed_iter);
                }
            }
        }
    }
}