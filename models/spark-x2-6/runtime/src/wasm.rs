use runtime::{error, json};
use spark_x2_6::session::{Engine, Message};
use wasm_bindgen::prelude::*;

runtime::runtime!(Engine);

#[wasm_bindgen]
impl Runtime {
    /// Starts replying to `message_json` (`{"text", "system"?, "thinking"?, "max_tokens"?,
    /// "sampling"?, "seed"}`), the conversation's next turn; returns its plan
    /// (`{"prompt_tokens", "max_tokens", "sampling"}`).
    pub fn start(&mut self, message_json: &str) -> Result<String, JsError> {
        let message: Message = serde_json::from_str(message_json)
            .map_err(|problem| error(spark_x2_6::Error::Request(problem.to_string())))?;
        Ok(json(&self.shell.engine()?.start(&message).map_err(error)?))
    }

    /// Runs the next stage; returns `{"stage", "reasoning", "answer", "tokens", "done"}`, the
    /// reply so far.
    pub fn step(&mut self) -> Result<String, JsError> {
        Ok(json(&self.shell.engine()?.step().map_err(error)?))
    }

    /// Forgets the conversation, to start a new one.
    pub fn reset(&mut self) -> Result<(), JsError> {
        self.shell.engine()?.reset();
        Ok(())
    }
}
