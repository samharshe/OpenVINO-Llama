use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use serde_json::Value;
use tokio::sync::oneshot;
use crate::model_config::ModelType;

type GenericError = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, GenericError>;
pub type BoxBody = http_body_util::combinators::BoxBody<Bytes, hyper::Error>;

pub fn full<T: Into<Bytes>>(chunk: T) -> BoxBody
{
    Full::new(chunk.into()).map_err(|never| match never {}).boxed()
}

#[derive(Debug)]
pub struct InferenceRequest
{
    pub data: Vec<u8>,
    pub model_type: ModelType,
    pub responder: oneshot::Sender<Value>,
}
