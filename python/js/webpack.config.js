var path = require("path");
var version = require("./package.json").version;
var TerserPlugin = require('terser-webpack-plugin');

// Custom webpack rules are generally the same for all webpack bundles, hence
// stored in a separate local variable.
var rules = [{test: /\.css$/, use: ["style-loader", "css-loader"]}];

module.exports = (env, argv) => {
    var devtool = argv.mode === "development" ? "source-map" : false;
    var isProduction = argv.mode === "production";
    
    // Common optimization settings
    var optimization = {
        minimize: isProduction,
        minimizer: isProduction ? [
            new TerserPlugin({
                terserOptions: {
                    compress: {
                        drop_console: true,
                        drop_debugger: true,
                        pure_funcs: ['console.log'],
                    },
                    mangle: true,
                    output: {
                        comments: false,
                    },
                },
                extractComments: false,
            })
        ] : [],
        sideEffects: false,
        usedExports: true,
        splitChunks: {
            chunks: 'all',
            minSize: 20000,
            maxSize: 250000,
            cacheGroups: {
                vendor: {
                    test: /[\\/]node_modules[\\/]/,
                    name: 'vendors',
                    chunks: 'all',
                    priority: 10,
                    enforce: true,
                },
            },
        },
    };
    
    return [
        {
            // Notebook extension
            //
            // This bundle only contains the part of the JavaScript that is run on
            // load of the notebook. This section generally only performs
            // some configuration for requirejs, and provides the legacy
            // "load_ipython_extension" function which is required for any notebook
            // extension.
            //
            entry: "./lib/extension.js",
            output: {
                filename: "extension.js",
                path: path.resolve(__dirname, "..", "cloudViewer", "nbextension"),
                libraryTarget: "amd",
                publicPath: "", // publicPath is set in extension.js
            },
            devtool,
        },
        {
            // Bundle for the notebook containing the custom widget views and models
            //
            // This bundle contains the implementation for the custom widget views and
            // custom widget.
            // It must be an amd module
            //
            entry: "./lib/index.js",
            output: {
                filename: "index.js",
                path: path.resolve(__dirname, "..", "cloudViewer", "nbextension"),
                libraryTarget: "amd",
                publicPath: "",
            },
            devtool,
            optimization,
            module: {
                rules: rules,
            },
            externals: ["@jupyter-widgets/base"],
        },
        {
            // Embeddable cloudViewer bundle
            //
            // This bundle is generally almost identical to the notebook bundle
            // containing the custom widget views and models.
            //
            // The only difference is in the configuration of the webpack public path
            // for the static assets.
            //
            // It will be automatically distributed by unpkg to work with the static
            // widget embedder.
            //
            // The target bundle is always `dist/index.js`, which is the path required
            // by the custom widget embedder.
            //
            entry: "./lib/embed.js",
            output: {
                filename: "index.js",
                path: path.resolve(__dirname, "dist"),
                libraryTarget: "amd",
                publicPath: "https://unpkg.com/cloudViewer@" + version + "/dist/",
            },
            devtool,
            optimization,
            module: {
                rules: rules,
            },
            externals: ["@jupyter-widgets/base"],
        },
    ];
};
