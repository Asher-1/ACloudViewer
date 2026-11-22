'use strict'
// A JavaScript action receives the runner's ArtifactService context directly;
// a shell step is not guaranteed to receive those credentials.
const {runCLI} = require('../../../docs/automation/scripts/wait_pages_artifact.cjs')
runCLI()
