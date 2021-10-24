def spec(x, y, color="run ID"):
    def subfigure(params, x_kwargs, y_kwargs):
        return {
            "height": 400,
            "width": 600,
            "encoding": {
                "x": {"type": "quantitative", "field": x, **x_kwargs},
                "y": {"type": "quantitative", "field": y, **y_kwargs},
                "color": {"type": "nominal", "field": color},
                "opacity": {
                    "value": 0.1,
                    "condition": {
                        "test": {
                            "and": [
                                {"param": "legend_selection"},
                                {"param": "hover"},
                            ]
                        },
                        "value": 1,
                    },
                },
            },
            "layer": [
                {
                    "mark": "line",
                    "params": params,
                }
            ],
        }

    params = [
        {
            "bind": "legend",
            "name": "legend_selection",
            "select": {
                "on": "mouseover",
                "type": "point",
                "fields": ["run ID"],
            },
        },
        {
            "bind": "legend",
            "name": "hover",
            "select": {
                "on": "mouseover",
                "type": "point",
                "fields": ["run ID"],
            },
        },
    ]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "data"},
        "transform": [{"filter": {"field": y, "valid": True}}],
        "hconcat": [
            subfigure(
                params=[*params, {"name": "selection", "select": "interval"}],
                x_kwargs={},
                y_kwargs={},
            ),
            subfigure(
                params=params,
                x_kwargs={"scale": {"domain": {"param": "selection", "encoding": "x"}}},
                y_kwargs={"scale": {"domain": {"param": "selection", "encoding": "y"}}},
            ),
        ],
    }
