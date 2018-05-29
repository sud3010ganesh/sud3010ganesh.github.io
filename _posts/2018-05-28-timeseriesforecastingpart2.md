---
title: "Time Series Forecasting - Part 2"
permalink: /2018-05-28-timeseriesforecastingpart2/
---

Autoregressive-Integrated-Moving Average (ARIMA) modelling is another popular technique for time series forecasting which aims to describe autocorrelations in the data. While exponential smoothing methods do not make any assumptions about correlations between successive values of the time series, in some cases you can make a better predictive model by taking correlations in the data into account. Autoregressive Integrated Moving Average (ARIMA) models include an explicit statistical model for the irregular component of a time series, that allows for non-zero autocorrelations in the irregular component
The process would be divided into three parts:
Phase I – Identify and Establish stationarity by examining ACF and PACF plots
Phase II – Estimate parameters in potential models and select best model based in evaluation metric
Phase III – Make forecast and Evaluate Model quality
