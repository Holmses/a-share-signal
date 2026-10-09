<script setup lang="ts">
import * as echarts from 'echarts'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

interface Point { date: string; value: number; drawdown?: number }
interface Series { id: string; name: string; data: Point[] }
interface Benchmark { code: string; name: string; data: Point[] }

const props = defineProps<{ series: Series[]; benchmark: Benchmark | null; mode: 'equity' | 'drawdown' | 'excess' }>()
const host = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null
let observer: ResizeObserver | null = null

function valuesFor(item: Series): Array<[string, number | null]> {
  if (props.mode === 'drawdown') return item.data.map((point) => [point.date, point.drawdown ?? null])
  if (props.mode === 'excess') {
    const benchmark = new Map((props.benchmark?.data ?? []).map((point) => [point.date, point.value]))
    return item.data.map((point) => [point.date, benchmark.has(point.date) ? point.value / Number(benchmark.get(point.date)) - 1 : null])
  }
  return item.data.map((point) => [point.date, point.value])
}

function render() {
  if (!host.value) return
  if (!chart) chart = echarts.init(host.value)
  const colors = ['#176b87', '#c43b32', '#7b5aa6', '#9a640e', '#16805b']
  const series: echarts.SeriesOption[] = props.series.map((item, index) => ({
    name: item.name,
    type: 'line',
    showSymbol: false,
    smooth: false,
    sampling: 'lttb',
    data: valuesFor(item),
    lineStyle: { width: index === 0 ? 2.2 : 1.6, color: colors[index % colors.length] },
    itemStyle: { color: colors[index % colors.length] },
  })) as echarts.SeriesOption[]
  if (props.mode === 'equity' && props.benchmark?.data.length) {
    series.push({
      name: props.benchmark.name,
      type: 'line', showSymbol: false, smooth: false, sampling: 'lttb',
      data: props.benchmark.data.map((point) => [point.date, point.value]),
      lineStyle: { width: 1.2, type: 'dashed', color: '#68747a' }, itemStyle: { color: '#68747a' },
    } as echarts.SeriesOption)
  }
  chart.setOption({
    animation: false,
    color: colors,
    legend: { top: 4, left: 8, type: 'scroll', textStyle: { color: '#66747b', fontSize: 11 } },
    tooltip: { trigger: 'axis', valueFormatter: (value: unknown) => typeof value === 'number' ? (props.mode === 'equity' ? value.toFixed(3) : `${(value * 100).toFixed(2)}%`) : '--' },
    grid: { left: 64, right: 24, top: 48, bottom: 54 },
    xAxis: { type: 'time', axisLine: { lineStyle: { color: '#cbd4d8' } }, axisLabel: { color: '#758187', fontSize: 10 } },
    yAxis: { type: 'value', scale: true, axisLabel: { color: '#758187', fontSize: 10, formatter: (value: number) => props.mode === 'equity' ? value.toFixed(2) : `${(value * 100).toFixed(0)}%` }, splitLine: { lineStyle: { color: '#edf1f2' } } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8, borderColor: '#dce3e6', fillerColor: 'rgba(23,107,135,.12)', handleStyle: { color: '#176b87' } }],
    series,
  }, true)
}

watch(() => [props.series, props.benchmark, props.mode], () => nextTick(render), { deep: true })
onMounted(() => { render(); if (host.value) { observer = new ResizeObserver(() => chart?.resize()); observer.observe(host.value) } })
onBeforeUnmount(() => { observer?.disconnect(); chart?.dispose() })
</script>

<template><div ref="host" class="chart-host compact" aria-label="策略净值、回撤或超额收益对比图"></div></template>
