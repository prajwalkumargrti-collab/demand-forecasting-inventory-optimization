"""Model Performance page for the dashboard."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.forecasting import get_forecast_bias


def render(
    metrics: dict,
    baseline_metrics: dict,
    test_actuals: np.ndarray,
    test_preds: np.ndarray,
    feature_importances: dict,
    df: pd.DataFrame,
) -> None:
    """Render the Model Performance page."""
    st.title("🎯 Model Performance")
    st.markdown("---")

    try:
        # ── KPI Row ───────────────────────────────────────────────────
        mae = metrics.get("MAE", metrics.get("mae", 0))
        rmse = metrics.get("RMSE", metrics.get("rmse", 0))
        mape = metrics.get("MAPE", metrics.get("mape", 0))

        k1, k2, k3 = st.columns(3)
        k1.metric("MAE", f"{mae:,.4f}")
        k2.metric("RMSE", f"{rmse:,.4f}")
        k3.metric("MAPE", f"{mape:,.2f}%")

        st.markdown("---")

        # ── Baseline Comparison ────────────────────────────────────────
        st.subheader("Model vs Baseline")

        bl_mae = baseline_metrics.get("MAE", baseline_metrics.get("mae", mae))
        bl_rmse = baseline_metrics.get("RMSE", baseline_metrics.get("rmse", rmse))
        bl_mape = baseline_metrics.get("MAPE", baseline_metrics.get("mape", mape))

        def pct_improvement(model_val, base_val):
            if base_val == 0:
                return 0.0
            return (base_val - model_val) / base_val * 100

        c1, c2, c3 = st.columns(3)
        with c1:
            imp = pct_improvement(mae, bl_mae)
            st.metric(
                "MAE",
                f"{mae:,.4f}",
                delta=f"{imp:+.1f}% vs baseline",
                delta_color="inverse",
            )
            st.caption(f"Baseline MAE: {bl_mae:,.4f}")
        with c2:
            imp = pct_improvement(rmse, bl_rmse)
            st.metric(
                "RMSE",
                f"{rmse:,.4f}",
                delta=f"{imp:+.1f}% vs baseline",
                delta_color="inverse",
            )
            st.caption(f"Baseline RMSE: {bl_rmse:,.4f}")
        with c3:
            imp = pct_improvement(mape, bl_mape)
            st.metric(
                "MAPE",
                f"{mape:,.2f}%",
                delta=f"{imp:+.1f}% vs baseline",
                delta_color="inverse",
            )
            st.caption(f"Baseline MAPE: {bl_mape:,.2f}%")

        st.markdown("---")

        # ── Actual vs Predicted Scatter ────────────────────────────────
        st.subheader("Actual vs Predicted")

        scatter_df = pd.DataFrame(
            {"Actual": test_actuals, "Predicted": test_preds}
        )

        fig_scatter = px.scatter(
            scatter_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Values",
            template="plotly_white",
            opacity=0.5,
        )
        all_vals = np.concatenate([test_actuals, test_preds])
        min_val, max_val = float(all_vals.min()), float(all_vals.max())
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="#e74c3c", dash="dash", width=2),
            )
        )
        fig_scatter.update_layout(
            xaxis_title="Actual Demand",
            yaxis_title="Predicted Demand",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Residual Distribution ──────────────────────────────────────
        st.subheader("Residual Distribution")

        residuals = test_actuals - test_preds

        fig_resid = px.histogram(
            x=residuals,
            nbins=50,
            title="Distribution of Residuals (Actual − Predicted)",
            template="plotly_white",
            labels={"x": "Residual", "y": "Count"},
        )
        fig_resid.update_layout(
            xaxis_title="Residual",
            yaxis_title="Frequency",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_resid, use_container_width=True)

        # ── Forecast Bias ──────────────────────────────────────────────
        st.subheader("Forecast Bias")

        try:
            bias = get_forecast_bias(test_actuals, test_preds)
        except Exception:
            bias = float(np.mean(residuals))

        bias_col1, bias_col2 = st.columns([1, 3])
        with bias_col1:
            st.metric("Bias", f"{bias:+.4f}")
        with bias_col2:
            if abs(bias) < 0.5:
                st.success(
                    "The model shows minimal bias — predictions are well-calibrated."
                )
            elif bias > 0:
                st.warning(
                    f"Positive bias ({bias:+.4f}): the model tends to **under-predict** demand. "
                    "Consider adjusting for higher safety stock."
                )
            else:
                st.warning(
                    f"Negative bias ({bias:+.4f}): the model tends to **over-predict** demand. "
                    "This may lead to excess inventory."
                )

        st.markdown("---")

        # ── Feature Importance ─────────────────────────────────────────
        st.subheader("Top Feature Importances")

        fi_df = (
            pd.DataFrame(
                list(feature_importances.items()),
                columns=["Feature", "Importance"],
            )
            .sort_values("Importance", ascending=True)
            .tail(15)
        )

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Features by Importance",
            template="plotly_white",
        )
        fig_fi.update_traces(marker_color="#2980b9")
        fig_fi.update_layout(
            xaxis_title="Importance",
            yaxis_title="",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── Error by Category ──────────────────────────────────────────
        st.subheader("MAE by Product Category")

        try:
            if "Category" in df.columns:
                n_test = len(test_actuals)
                test_df = df.tail(n_test).copy()
                test_df = test_df.reset_index(drop=True)
                test_df["Residual_Abs"] = np.abs(
                    test_actuals[: len(test_df)] - test_preds[: len(test_df)]
                )

                fig_box = px.box(
                    test_df,
                    x="Category",
                    y="Residual_Abs",
                    title="Absolute Error Distribution by Category",
                    template="plotly_white",
                    color="Category",
                )
                fig_box.update_layout(
                    showlegend=False,
                    xaxis_title="Category",
                    yaxis_title="Absolute Error",
                    margin=dict(t=50, b=40),
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("Category column not found — skipping per-category error breakdown.")
        except Exception as cat_err:
            st.warning(f"Could not compute per-category error: {cat_err}")

    except Exception as e:
        st.error(f"Error rendering Model Performance: {e}")
