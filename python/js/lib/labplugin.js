var plugin = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
  id: 'cloudViewer:plugin',
  requires: [base.IJupyterWidgetRegistry],
  activate: (app, widgets) => {
    widgets.registerWidget({
      name: 'cloudViewer',
      version: plugin.version,
      exports: plugin
    });
  },
  autoStart: true
};
