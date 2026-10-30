//! Frontends: one per source model format.
//!
//! A frontend knows how to parse its format into a [`SourceGraph`] and owns
//! the [`LoweringRegistry`] that maps the format's operators onto TOSA. The
//! [`FrontendSet`] picks a frontend by name, file extension or by sniffing
//! the file's contents.

use std::path::Path;

use crate::error::{ConvertError, Result};
use crate::lower::LoweringRegistry;
use crate::source::SourceGraph;

pub trait Frontend: Send + Sync {
    /// Short identifier, e.g. `"tflite"`.
    fn name(&self) -> &str;

    /// File extensions (without the dot) this frontend handles.
    fn extensions(&self) -> &[&str];

    /// Whether `bytes` look like a model in this format (magic numbers,
    /// file identifiers). Used when the file extension is not telling.
    fn sniff(&self, _bytes: &[u8]) -> bool {
        false
    }

    /// Parse a model.
    fn load(&self, bytes: &[u8]) -> Result<SourceGraph>;

    /// The lowerings for this format's operators.
    fn lowerings(&self) -> &LoweringRegistry;
}

/// The frontends available to a tool, searched in registration order.
#[derive(Default)]
pub struct FrontendSet {
    frontends: Vec<Box<dyn Frontend>>,
}

impl FrontendSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, frontend: impl Frontend + 'static) -> &mut Self {
        self.frontends.push(Box::new(frontend));
        self
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn Frontend> {
        self.frontends.iter().map(Box::as_ref)
    }

    pub fn names(&self) -> Vec<&str> {
        self.iter().map(Frontend::name).collect()
    }

    pub fn by_name(&self, name: &str) -> Result<&dyn Frontend> {
        self.iter().find(|f| f.name() == name).ok_or_else(|| ConvertError::NoFrontend(format!("format '{name}'")))
    }

    /// The frontend for `path`, by its extension.
    pub fn for_path(&self, path: &Path) -> Result<&dyn Frontend> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        self.iter()
            .find(|f| f.extensions().iter().any(|e| e.eq_ignore_ascii_case(&ext)))
            .ok_or_else(|| ConvertError::NoFrontend(format!("files with extension '.{ext}'")))
    }

    /// The frontend that recognises the contents of `bytes`.
    pub fn sniff(&self, bytes: &[u8]) -> Result<&dyn Frontend> {
        self.iter().find(|f| f.sniff(bytes)).ok_or_else(|| ConvertError::NoFrontend("this file's contents".into()))
    }

    /// The frontend for a file: by explicit name, else by contents, else by
    /// extension.
    pub fn select(&self, name: Option<&str>, path: &Path, bytes: &[u8]) -> Result<&dyn Frontend> {
        match name {
            Some(n) => self.by_name(n),
            None => self.sniff(bytes).or_else(|_| self.for_path(path)),
        }
    }
}
