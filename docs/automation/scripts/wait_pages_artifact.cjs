'use strict'

class VisibilityError extends Error {}

// Wait on the same runtime ArtifactService used by deploy-pages, before it
// creates any deployment. A successful REST lookup is not an equivalent gate.
async function waitPagesArtifact({client, name, expectedId, timeoutMs = 90000,
  now = () => performance.now(), sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
  log = message => console.log(message)}) {
  if (typeof name !== 'string' || !name || !/^[1-9]\d*$/.test(String(expectedId)) ||
      !Number.isSafeInteger(Number(expectedId)) || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new VisibilityError('Invalid expected Pages artifact name, ID or timeout')
  }
  const id = Number(expectedId)
  const deadline = now() + timeoutMs
  let delay = 1000
  let attempts = 0
  while (true) {
    // Transport/auth/service errors propagate. Only a successful, valid list
    // that has not observed our finalized artifact is eligible for retry.
    let timer
    let response
    try {
      response = await Promise.race([
        client.listArtifacts(),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new VisibilityError('Pages artifact visibility deadline exceeded')),
            Math.max(0, deadline - now()))
        })
      ])
    } finally {
      clearTimeout(timer)
    }
    attempts++
    if (!response || !Array.isArray(response.artifacts) || response.artifacts.some(
      artifact => !artifact || typeof artifact.name !== 'string' ||
        !Number.isSafeInteger(artifact.id) || artifact.id <= 0 ||
        !Number.isFinite(artifact.size) || artifact.size < 0)) {
      throw new VisibilityError('Invalid runtime artifact inventory')
    }
    const matches = response.artifacts.filter(artifact => artifact.name === name)
    if (matches.length > 1) throw new VisibilityError('Multiple runtime artifacts match the Pages name')
    const wrongName = response.artifacts.find(artifact => artifact.id === id && artifact.name !== name)
    if (wrongName || (matches.length === 1 && matches[0].id !== id)) {
      throw new VisibilityError('Runtime Pages artifact name/ID does not match the upload receipt')
    }
    if (matches.length === 1) {
      if (matches[0].size <= 0) throw new VisibilityError('Runtime Pages artifact has no content')
      if (now() >= deadline) throw new VisibilityError('Pages artifact visibility deadline exceeded')
      log(`Pages artifact ${name} (${id}) visible through runtime service after ${attempts} list(s)`)
      return matches[0]
    }
    const remaining = deadline - now()
    if (remaining <= 0) throw new VisibilityError('Pages artifact visibility deadline exceeded')
    log(`Pages artifact ${name} (${id}) not yet visible; runtime list attempt ${attempts}`)
    await sleep(Math.min(delay, remaining))
    if (now() >= deadline) throw new VisibilityError('Pages artifact visibility deadline exceeded')
    delay = Math.min(delay * 2, 5000)
  }
}

async function runCLI() {
  try {
    if (!process.env.ACTIONS_RUNTIME_TOKEN || !process.env.ACTIONS_RESULTS_URL) {
      throw new VisibilityError('Artifact runtime context is missing; use the JavaScript action entry point')
    }
    const {DefaultArtifactClient} = require('@actions/artifact')
    await waitPagesArtifact({client: new DefaultArtifactClient(),
      name: process.env.PAGES_ARTIFACT_NAME, expectedId: process.env.PAGES_ARTIFACT_ID})
  } catch (error) {
    // Never print raw service exceptions or authenticated runtime URLs.
    console.error(error instanceof VisibilityError ? error.message :
      'Runtime artifact service request failed; visibility was not established')
    // The SDK has no cancellation API; close this dedicated action process so
    // a hung socket cannot extend the bounded visibility deadline.
    process.exit(1)
  }
}

module.exports = {waitPagesArtifact, runCLI}
if (require.main === module) runCLI()
