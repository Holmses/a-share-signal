import { createRouter, createWebHistory } from 'vue-router'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'dashboard', component: () => import('./views/DashboardView.vue') },
    { path: '/stock/:symbol?', name: 'stock', component: () => import('./views/StockView.vue') },
    { path: '/backtest', name: 'backtest', component: () => import('./views/BacktestView.vue') },
    { path: '/compare', name: 'compare', component: () => import('./views/CompareView.vue') },
    { path: '/tasks', name: 'tasks', component: () => import('./views/TasksView.vue') },
  ],
})
