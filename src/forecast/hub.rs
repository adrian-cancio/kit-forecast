use std::path::PathBuf;

use crate::{Error, Result};

pub(crate) fn download(repo_id: &str, file: &str) -> Result<PathBuf> {
    let (owner, name) = repo_id.split_once('/').ok_or_else(|| {
        Error::InvalidConfig(format!(
            "repository id {repo_id:?} must look like \"owner/name\""
        ))
    })?;
    let client = hf_hub::HFClientSync::new()
        .map_err(|e| Error::Download(format!("cannot create Hugging Face client: {e}")))?;
    client
        .model(owner, name)
        .download_file()
        .filename(file)
        .send()
        .map_err(|e| Error::Download(format!("{repo_id}/{file}: {e}")))
}
