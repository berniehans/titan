/// SSE framing helpers and event stream construction.
///
/// Converts the engine's typed token chunks into `text/event-stream` wire
/// frames (`data: {json}\n\n`) and terminates the feed with the standard
/// `data: [DONE]\n\n` marker expected by OpenAI-compatible clients.
use axum::response::sse::{Event, Sse};
use futures_core::stream::Stream;
use futures_util::stream::{self, StreamExt};

/// Formats a single-line SSE `data:` frame terminated by a blank line.
pub fn chunk_frame(json: &str) -> String {
    format!("data: {json}\n\n")
}

/// The terminal SSE frame in OpenAI streaming responses.
pub fn done_frame() -> &'static str {
    "data: [DONE]\n\n"
}

/// Builds an [`Event`] whose `data:` line carries `payload`.
pub fn data_event(payload: impl AsRef<str>) -> Event {
    Event::default().data(payload)
}

/// Builds the terminal `[DONE]` event.
pub fn done_event() -> Event {
    Event::default().data("[DONE]")
}

/// Assembles a `Sse` response streaming `events`.
///
/// The event vector is lifted into the `Ok<Event>` TryStream via the standard
/// map adapter so axum can serve it as a finite `text/event-stream` body.
pub fn to_sse(
    events: Vec<Event>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    Sse::new(stream::iter(events).map(Ok))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_frame_is_single_data_line_with_blank_terminal_line() {
        assert_eq!(chunk_frame(r#"{"t":1}"#), "data: {\"t\":1}\n\n");
    }

    #[test]
    fn done_frame_is_terminal_marker() {
        assert_eq!(done_frame(), "data: [DONE]\n\n");
    }

    #[test]
    fn event_builders_are_constructible() {
        let _middle = data_event(r#"{"t":0}"#);
        let _terminal = done_event();
        assert_eq!(done_frame(), "data: [DONE]\n\n");
    }
}
