pub mod association;
pub mod cached_path;
pub mod short_only;
pub mod similarity;

pub trait RetrStrategy {
    type Request: RetrRequest; //接受的查询参数类型
    type Return<'a>
    where
        Self: 'a;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_>;
}

pub trait RetrRequest {}
