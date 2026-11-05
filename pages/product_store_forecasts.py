"""Product / Store Forecasts page for the dashboard."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.forecasting import generate_forecast


def render(
    df: pd.DataFrame,
    model,
    test_preds: np.ndarray,
    test_actuals: np.ndarray,
) -> None:
    """Render the Product/Store Forecasts page."""
    st.title("🔮 Product/Store Forecasts")
    st.markdown("---")

    try:
        stores = sorted(df["Store ID"].unique())
        products = sorted(df["Product ID"].unique())

        col_s, col_p = st.columns(2)
        with col_s:
            selected_store = st.selectbox("Select Store", stores, index=0)
        with col_p:
            selected_product = st.selectbox("Select Product", products, index=0)

        subset = df[
            (df["Store ID"] == selected_store) & (df["Product ID"] == selected_product)
        ].copy()
        subset = subset.sort_values("Date")

        if subset.empty:
            st.warning("No data available for the selected Store / Product combination.")
            return

        # ── Chart 1: Historical demand trajectory ──────────────────────
        st.subheader("Historical Demand Trajectory")
        fig_hist = px.line(
            subset,
            x="Date",
            y="Demand",
            title=f"Demand Over Time — {selected_store} / {selected_product}",
            template="plotly_white",
        )
        fig_hist.update_traces(line=dict(color="#1f4e79", width=2))
        fig_hist.update_layout(
            xaxis_title="Date",
            yaxis_title="Demand",
            hovermode="x unified",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Chart 2: Actual vs Forecast on the test period ─────────────
        st.subheader("Actual vs Forecast (Test Period)")

        test_df = getattr(model, "test_df", None)
        has_test_overlay = False

        if test_df is not None:
            sp_test = test_df[
                (test_df["Store ID"] == selected_store) & (test_df["Product ID"] == selected_product)
            ].sort_values("Date")
            if not sp_test.empty:
                test_dates = sp_test["Date"].values
                actual_series = sp_test["Demand"].values
                pred_series = sp_test["Predicted"].values
                has_test_overlay = True

        if has_test_overlay:
            residuals = actual_series - pred_series
            residual_std = float(np.std(residuals)) if len(residuals) > 1 else 5.0

            fig_av = go.Figure()

            # Full historical actuals
            fig_av.add_trace(
                go.Scatter(
                    x=subset["Date"],
                    y=subset["Demand"],
                    mode="lines",
                    name="Actual",
                    line=dict(color="#1f4e79", width=2),
                )
            )

            # Forecast line on test dates
            fig_av.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=pred_series,
                    mode="lines",
                    name="Forecast",
                    line=dict(color="#e74c3c", width=2, dash="dash"),
                )
            )

            # Confidence band (±1.96σ)
            upper = pred_series + 1.96 * residual_std
            lower = np.maximum(0, pred_series - 1.96 * residual_std)

            fig_av.add_trace(
                go.Scatter(
                    x=np.concatenate([test_dates, test_dates[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(231,76,60,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="95 % Confidence",
                )
            )

            fig_av.update_layout(
                title=f"Actual vs Forecast — {selected_store} / {selected_product}",
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Demand",
                hovermode="x unified",
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig_av, use_container_width=True)
        else:
            st.info(
                "Test period overlay is being generated..."
            )

        # ── Chart 3: 30-day forward forecast ──────────────────────────
        st.subheader("30-Day Forward Forecast")
        try:
            forecast_df = generate_forecast(model, subset, horizon_days=30)

            if forecast_df is not None and not forecast_df.empty:
                forecast_col = (
                    "Forecast" if "Forecast" in forecast_df.columns else forecast_df.columns[-1]
                )
                date_col = (
                    "Date" if "Date" in forecast_df.columns else forecast_df.columns[0]
                )

                fc_values = forecast_df[forecast_col].values
                fc_dates = forecast_df[date_col].values

                if n_test > 0:
                    res_std = float(np.std(test_actuals - test_preds))
                else:
                    res_std = float(fc_values.std()) * 0.2

                upper_fc = fc_values + 1.96 * res_std
                lower_fc = fc_values - 1.96 * res_std

                fig_fc = go.Figure()
                fig_fc.add_trace(
                    go.Scatter(
                        x=fc_dates,
                        y=fc_values,
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#2ecc71", width=2),
                        marker=dict(size=4),
                    )
                )
                fig_fc.add_trace(
                    go.Scatter(
                        x=np.concatenate([fc_dates, fc_dates[::-1]]),
                        y=np.concatenate([upper_fc, lower_fc[::-1]]),
                        fill="toself",
                        fillcolor="rgba(46,204,113,0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=True,
                        name="95 % Confidence",
                    )
                )
                fig_fc.update_layout(
                    title=f"30-Day Forecast — {selected_store} / {selected_product}",
                    template="plotly_white",
                    xaxis_title="Date",
                    yaxis_title="Forecasted Demand",
                    hovermode="x unified",
                    margin=dict(t=50, b=40),
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                st.subheader("Forecast Values (Next 30 Days)")
                st.dataframe(
                    forecast_df.reset_index(drop=True).style.format(
                        {forecast_col: "{:.2f}"}
                    ),
                    use_container_width=True,
                )
            else:
                st.warning("Forecast generation returned no data.")
        except Exception as fc_err:
            st.error(f"Could not generate 30-day forecast: {fc_err}")

    except Exception as e:
        st.error(f"Error rendering Product/Store Forecasts: {e}")
