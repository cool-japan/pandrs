use std::collections::HashMap;
use std::sync::Arc;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;

use super::registry::PluginRegistry;

/// A single step in a plugin pipeline
#[derive(Debug, Clone)]
pub enum PipelineStep {
    Source {
        plugin_name: String,
        options: HashMap<String, String>,
    },
    Transform {
        plugin_name: String,
        options: HashMap<String, String>,
    },
    Validate {
        plugin_name: String,
        options: HashMap<String, String>,
    },
    Sink {
        plugin_name: String,
        options: HashMap<String, String>,
    },
}

impl PipelineStep {
    pub fn plugin_name(&self) -> &str {
        match self {
            PipelineStep::Source { plugin_name, .. } => plugin_name,
            PipelineStep::Transform { plugin_name, .. } => plugin_name,
            PipelineStep::Validate { plugin_name, .. } => plugin_name,
            PipelineStep::Sink { plugin_name, .. } => plugin_name,
        }
    }
}

/// A composable pipeline of plugin steps
///
/// A pipeline must start with exactly one Source step and may be followed by
/// any number of Transform / Validate steps, optionally ending with a Sink.
pub struct PluginPipeline {
    registry: Arc<PluginRegistry>,
    steps: Vec<PipelineStep>,
}

impl PluginPipeline {
    /// Create a new empty pipeline backed by the given registry
    pub fn new(registry: Arc<PluginRegistry>) -> Self {
        PluginPipeline {
            registry,
            steps: Vec::new(),
        }
    }

    /// Append a source step (reads data from an external source)
    pub fn source(mut self, plugin_name: &str, options: HashMap<String, String>) -> Self {
        self.steps.push(PipelineStep::Source {
            plugin_name: plugin_name.to_string(),
            options,
        });
        self
    }

    /// Append a transform step
    pub fn transform(mut self, plugin_name: &str, options: HashMap<String, String>) -> Self {
        self.steps.push(PipelineStep::Transform {
            plugin_name: plugin_name.to_string(),
            options,
        });
        self
    }

    /// Append a validate step (runs validation, continues the pipeline on success)
    pub fn validate(mut self, plugin_name: &str, options: HashMap<String, String>) -> Self {
        self.steps.push(PipelineStep::Validate {
            plugin_name: plugin_name.to_string(),
            options,
        });
        self
    }

    /// Append a sink step (writes data to an external destination)
    pub fn sink(mut self, plugin_name: &str, options: HashMap<String, String>) -> Self {
        self.steps.push(PipelineStep::Sink {
            plugin_name: plugin_name.to_string(),
            options,
        });
        self
    }

    /// Execute the pipeline.
    ///
    /// Returns `Ok(Some(DataFrame))` if the pipeline ends without a sink (i.e. the
    /// last step is a Source, Transform, or Validate step).
    /// Returns `Ok(None)` if the pipeline ends with a Sink step.
    pub fn execute(&self) -> Result<Option<DataFrame>> {
        if self.steps.is_empty() {
            return Err(Error::InvalidOperation("Pipeline has no steps".to_string()));
        }

        // The first step must be a Source
        let first = self
            .steps
            .first()
            .ok_or_else(|| Error::InvalidOperation("Pipeline has no steps".to_string()))?;

        let mut current_df = match first {
            PipelineStep::Source {
                plugin_name,
                options,
            } => {
                let source = self.registry.get_source(plugin_name).ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "Pipeline: source plugin '{}' is not registered",
                        plugin_name
                    ))
                })?;
                source.read(options)?
            }
            other => {
                return Err(Error::InvalidOperation(format!(
                    "Pipeline must start with a Source step, found {:?}",
                    other.plugin_name()
                )))
            }
        };

        // Process remaining steps
        for step in self.steps.iter().skip(1) {
            match step {
                PipelineStep::Source { .. } => {
                    return Err(Error::InvalidOperation(
                        "Pipeline cannot have a Source step after the first step".to_string(),
                    ));
                }
                PipelineStep::Transform {
                    plugin_name,
                    options,
                } => {
                    let transform = self.registry.get_transform(plugin_name).ok_or_else(|| {
                        Error::InvalidInput(format!(
                            "Pipeline: transform plugin '{}' is not registered",
                            plugin_name
                        ))
                    })?;
                    current_df = transform.transform(current_df, options)?;
                }
                PipelineStep::Validate {
                    plugin_name,
                    options,
                } => {
                    let validator = self.registry.get_validator(plugin_name).ok_or_else(|| {
                        Error::InvalidInput(format!(
                            "Pipeline: validator plugin '{}' is not registered",
                            plugin_name
                        ))
                    })?;
                    let issues = validator.validate(&current_df, options)?;
                    // Fail-fast on any Error-severity issue
                    for issue in &issues {
                        if issue.severity == crate::plugins::traits::IssueSeverity::Error {
                            return Err(Error::InvalidOperation(format!(
                                "Validation failed: {}",
                                issue.message
                            )));
                        }
                    }
                }
                PipelineStep::Sink {
                    plugin_name,
                    options,
                } => {
                    let sink = self.registry.get_sink(plugin_name).ok_or_else(|| {
                        Error::InvalidInput(format!(
                            "Pipeline: sink plugin '{}' is not registered",
                            plugin_name
                        ))
                    })?;
                    sink.write(&current_df, options)?;
                    return Ok(None);
                }
            }
        }

        Ok(Some(current_df))
    }

    /// Returns the number of steps in the pipeline
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Returns a slice of pipeline steps for inspection
    pub fn steps(&self) -> &[PipelineStep] {
        &self.steps
    }
}
