<script setup lang="ts">
import * as echarts from 'echarts'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { Bar, TradeEvent } from '../types'

const props = defineProps<{ bars: Bar[]; events: TradeEvent[]; focusDate?: string }>()
const host = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null
let resizeObserver: ResizeObserver | null = null

function eventSeries(kind: TradeEvent['kind'], action?: TradeEvent['action']) {
  const values = props.events.filter((event) => event.kind === kind && (!action || event.action === action) && event.price != null)
  const labels: Record<string, string> = { signal: '策略信号', execution: '实际成交', pending: '待执行', unfilled: '未成交', rejected: '候选拒绝' }
  const isBuy = action === 'BUY'
  return {
    name: action ? `${labels[kind]}·${isBuy ? '买' : '卖'}` : labels[kind],
    type: 'scatter',
    xAxisIndex: 0,
    yAxisIndex: 0,
    symbol: kind === 'signal' ? 'emptyCircle' : kind === 'pending' ? 'diamond' : kind === 'unfilled' ? 'roundRect' : kind === 'rejected' ? 'rect' : 'triangle',
    symbolRotate: !isBuy && kind === 'execution' ? 180 : 0,
    symbolSize: kind === 'execution' ? 14 : 11,
    itemStyle: {
      color: kind === 'signal' ? '#176b87' : kind === 'pending' ? '#9a640e' : kind === 'unfilled' ? '#a13d37' : kind === 'rejected' ? '#6b7478' : isBuy ? '#c43b32' : '#16805b',
      borderWidth: 2,
    },
    data: values.map((event) => ({
      value: [event.date, event.price],
      event,
    })),
    z: 12,
  }
}

function initialZoom(dates: string[]): { start: number; end: number } {
  const focusDate = props.focusDate?.replaceAll('-', '')
  const defaultZoom = {
    start: dates.length > 252 ? Math.max(0, 100 - (252 / dates.length) * 100) : 0,
    end: 100,
  }
  if (!focusDate) return defaultZoom
  const focusIndex = dates.indexOf(focusDate)
  if (focusIndex < 0) {
    return defaultZoom
  }
  const priorBuy = props.events
    .filter((event) => event.kind === 'execution' && event.action === 'BUY' && event.date <= focusDate)
    .sort((left, right) => right.date.localeCompare(left.date))[0]
  const buyIndex = priorBuy ? dates.indexOf(priorBuy.date) : -1
  const leftIndex = Math.max(0, (buyIndex >= 0 ? buyIndex : focusIndex) - 20)
  const rightIndex = Math.min(dates.length - 1, focusIndex + 20)
  const denominator = Math.max(1, dates.length - 1)
  return {
    start: (leftIndex / denominator) * 100,
    end: (rightIndex / denominator) * 100,
  }
}

function render() {
  if (!host.value) return
  if (!chart) chart = echarts.init(host.value, undefined, { renderer: 'canvas' })
  const dates = props.bars.map((bar) => bar.trade_date)
  const zoom = initialZoom(dates)
  chart.setOption({
    animation: false,
    backgroundColor: '#ffffff',
    legend: { top: 8, left: 12, itemWidth: 18, itemHeight: 8, textStyle: { color: '#66747b', fontSize: 11 } },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross', lineStyle: { color: '#9caaaf' } },
      formatter(params: unknown) {
        const items = params as Array<{ seriesName: string; data: unknown; value: unknown }>
        if (!items.length) return ''
        const first = items[0]
        const date = Array.isArray(first.value) ? String(first.value[0]) : dates[(first as unknown as { dataIndex: number }).dataIndex]
        const bar = props.bars.find((value) => value.trade_date === date)
        const lines = [`<strong>${date}</strong>`]
        if (bar) lines.push(`开 ${bar.open.toFixed(2)} | 高 ${bar.high.toFixed(2)} | 低 ${bar.low.toFixed(2)} | 收 ${bar.close.toFixed(2)}`)
        for (const item of items) {
          const data = item.data as { event?: TradeEvent }
          if (data?.event) {
            const event = data.event
            lines.push(`<br><strong>${item.seriesName}</strong> ${event.price?.toFixed(2) ?? '--'} · ${event.quantity ?? '--'} 股`)
            lines.push(event.reason || '')
            if (event.delayed) lines.push('执行状态：延迟成交')
            if (event.kind === 'execution' && event.signal_date) lines.push(`信号日期：${event.signal_date}`)
            if (event.kind !== 'execution' && event.execution_date) lines.push(`对应成交日：${event.execution_date}`)
            if (event.market_state) lines.push(`市场状态：${event.market_state}`)
            if (event.pnl != null) lines.push(`单笔盈亏：${event.pnl.toFixed(2)}`)
          }
        }
        return lines.join('<br>')
      },
    },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    grid: [
      { left: 64, right: 24, top: 50, height: '62%' },
      { left: 64, right: 24, top: '76%', height: '14%' },
    ],
    xAxis: [
      { type: 'category', data: dates, boundaryGap: true, axisLine: { lineStyle: { color: '#cbd4d8' } }, axisLabel: { color: '#758187', fontSize: 10 }, splitLine: { show: false }, min: 'dataMin', max: 'dataMax' },
      { type: 'category', gridIndex: 1, data: dates, boundaryGap: true, axisLine: { lineStyle: { color: '#cbd4d8' } }, axisLabel: { color: '#758187', fontSize: 10 }, splitLine: { show: false }, min: 'dataMin', max: 'dataMax' },
    ],
    yAxis: [
      { scale: true, axisLabel: { color: '#758187', fontSize: 10 }, splitLine: { lineStyle: { color: '#edf1f2' } } },
      { scale: true, gridIndex: 1, axisLabel: { color: '#758187', fontSize: 10 }, splitLine: { show: false } },
    ],
    dataZoom: [
      { type: 'inside', xAxisIndex: [0, 1], start: zoom.start, end: zoom.end },
      { type: 'slider', xAxisIndex: [0, 1], bottom: 2, height: 18, start: zoom.start, end: zoom.end, borderColor: '#dce3e6', fillerColor: 'rgba(23,107,135,.12)', handleStyle: { color: '#176b87' } },
    ],
    series: [
      {
        name: '日 K', type: 'candlestick', data: props.bars.map((bar) => [bar.open, bar.close, bar.low, bar.high]),
        itemStyle: { color: '#c43b32', color0: '#16805b', borderColor: '#c43b32', borderColor0: '#16805b' },
      },
      ...([5, 10, 20, 60] as const).map((window, index) => ({
        name: `MA${window}`, type: 'line', showSymbol: false, smooth: false,
        data: props.bars.map((bar) => bar[`ma${window}`]),
        lineStyle: { width: 1, color: ['#d08a1d', '#176b87', '#7b5aa6', '#5d6b73'][index] },
      })),
      eventSeries('signal', 'BUY'),
      eventSeries('signal', 'SELL'),
      eventSeries('execution', 'BUY'),
      eventSeries('execution', 'SELL'),
      eventSeries('pending', 'BUY'),
      eventSeries('pending', 'SELL'),
      eventSeries('unfilled', 'BUY'),
      eventSeries('unfilled', 'SELL'),
      eventSeries('rejected', 'BUY'),
      {
        name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
        data: props.bars.map((bar, index) => ({ value: bar.volume, itemStyle: { color: index > 0 && bar.close >= props.bars[index - 1].close ? '#d46861' : '#4f9b7e' } })),
      },
    ],
  }, true)
}

function exportPng() {
  if (!chart) return null
  return chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#ffffff' })
}

defineExpose({ exportPng })

watch(() => [props.bars, props.events, props.focusDate], () => nextTick(render), { deep: true })
onMounted(() => {
  render()
  if (host.value) {
    resizeObserver = new ResizeObserver(() => chart?.resize())
    resizeObserver.observe(host.value)
  }
})
onBeforeUnmount(() => { resizeObserver?.disconnect(); chart?.dispose() })
</script>

<template><div ref="host" class="chart-host" aria-label="个股日K、均线、成交量与买卖点图表"></div></template>
