import { useState, useRef, useEffect } from 'react';
import { 
  FileDown, Activity, BrainCircuit, ShieldAlert, SunMoon, Moon,
  Wind, Droplets, Thermometer, CloudRain, Monitor, Smartphone, MapPin, 
  User, CalendarDays
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  BarChart, Bar, AreaChart, Area, ScatterChart, Scatter, Legend
} from 'recharts';
import html2pdf from 'html2pdf.js';
import rawData from './data.json';

const data = rawData;

// === DATA PREP ===
const hourlyCompareData = data.compare_hourly.hours.map((h: number, i: number) => ({
  hour: `${h}h`,
  INMET_PM25: data.compare_hourly.inmet_pm25[i],
  IQAIR_AQI: data.compare_hourly.iqair_aqi[i],
}));

const monthlyInmetData = data.inmet.monthly.labels.map((m: string, i: number) => ({
  month: m,
  Media: data.inmet.monthly.mean[i],
  Pico: data.inmet.monthly.p75[i],
}));

const rainEffectData = data.inmet.rain_effect.labels.map((l: string, i: number) => ({
  label: l,
  PM25: data.inmet.rain_effect.pm25[i],
}));

const modelData = data.model.sample_real.slice(0, 30).map((r: number, i: number) => ({
  id: i,
  Real: r,
  Previsto: data.model.sample_predicted[i],
}));


const scatterData = data.inmet.scatter_pm25_no2.no2.map((val: number, i: number) => ({
  no2: val,
  pm25: data.inmet.scatter_pm25_no2.pm25[i]
}));

const histogramData = data.inmet.histogram.bins.map((bin: number, i: number) => ({
  bin: bin.toFixed(1),
  Chuva: data.inmet.histogram.data.Chuva[i] || 0,
  Seca: data.inmet.histogram.data.Seca[i] || 0,
  Transicao: data.inmet.histogram.data["Transição"][i] || 0
})).slice(0, 20);

const dailyBySensorData = data.iqair.daily_by_sensor.Brasilia.dates.map((date: string, i: number) => ({
  date: date.substring(5),
  Brasilia: data.iqair.daily_by_sensor.Brasilia.aqi[i],
  Escola: data.iqair.daily_by_sensor["Escola 115 Norte"].aqi[i],
  Finatec: data.iqair.daily_by_sensor.Finatec.aqi[i],
  Gama: data.iqair.daily_by_sensor["Unb Odisseia Gama"].aqi[i]
}));

// Feature Importance
const featureImportanceData = data.model.features.map((f: string, i: number) => ({
  name: f.replace('pm25_','PM25 ').replace('rain_','Chuva ').replace('hour_','Hora ').replace('month_','Mês ').replace('_',' '),
  importance: +(data.model.feature_importance[i] * 100).toFixed(2)
})).sort((a: any, b: any) => b.importance - a.importance);

// Boxplot data por sensor
const boxPlotData = data.iqair.sensors.map((s: string) => {
  const box = (data.iqair.aqi_box_by_sensor as Record<string, {min:number;q1:number;median:number;q3:number;max:number;mean:number}>)[s];
  return {
    sensor: s.replace('Unb Odisseia ',''),
    min: box.min, q1: box.q1, median: box.median, q3: box.q3, max: box.max,
    mean: +box.mean.toFixed(1),
  };
});

// % acima OMS por mes
const omsData = data.inmet.pct_above_oms_15.map((v: number, i: number) => ({
  month: ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'][i],
  pct: v
}));

// Correlação
const corrLabels = data.inmet.correlation.labels;
const corrMatrix = data.inmet.correlation.matrix;

// Hourly by season
const hourlyBySeasonData2 = data.inmet.hourly_by_season.Chuva.map((val: number, i: number) => ({
  hour: `${i}h`,
  Chuva: val,
  Seca: data.inmet.hourly_by_season.Seca[i],
  Transicao: data.inmet.hourly_by_season["Transição"][i]
}));

// Sensor stats table
const sensorStatsData = data.iqair.sensor_stats;

// LOGIC FOR MOBILE APP
const aqi = data.iqair.aqi_mean;
const temp = data.iqair.temp_mean;
const isFavorable = aqi < 20 && temp < 30;

export default function App() {
  const [viewMode, setViewMode] = useState<'desktop' | 'mobile'>('mobile');
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [activeTab, setActiveTab] = useState('home');
  const [isExporting, setIsExporting] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const reportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.body.className = theme;
  }, [theme]);

  const exportPDF = async () => {
    setIsExporting(true);
    setShowReport(true);
    await new Promise(r => setTimeout(r, 1200));
    
    const element = reportRef.current;
    if (!element) { setIsExporting(false); setShowReport(false); return; }

    const opt = {
      margin: [15, 15, 15, 15] as [number, number, number, number],
      filename: 'MotoAR_Relatorio_Completo.pdf',
      image: { type: 'jpeg' as const, quality: 0.95 },
      html2canvas: { scale: 2, useCORS: true, logging: false, backgroundColor: '#ffffff' },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' as const },
      pagebreak: { mode: ['avoid-all', 'css', 'legacy'] }
    };

    try {
      await html2pdf().from(element).set(opt).save();
    } catch (err) {
      console.error(err);
      alert("Erro ao exportar PDF.");
    } finally {
      setShowReport(false);
      setIsExporting(false);
    }
  };

  const getCssVar = (name: string) => `var(${name})`;
  const chartProps = {
    cartesianStroke: "var(--border)",
    axisStroke: "var(--txt2)",
    tooltipBg: "var(--card)",
    tooltipBorder: "1px solid var(--border)",
  };
  const TooltipStyle = { background: chartProps.tooltipBg, border: chartProps.tooltipBorder, borderRadius: 8, color: 'var(--txt)' };

  // --- CHART RENDER FUNCTIONS ---
  const renderChart1 = () => (
    <div className="card">
      <div className="card-title">Poluição Hoje (IQAir vs INMET)</div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={hourlyCompareData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartProps.cartesianStroke} vertical={false} />
            <XAxis dataKey="hour" stroke={chartProps.axisStroke} fontSize={12} fontWeight={700} />
            <YAxis stroke={chartProps.axisStroke} fontSize={12} />
            <RechartsTooltip contentStyle={TooltipStyle} />
            <Legend wrapperStyle={{fontWeight: 700, fontSize: 12, color: 'var(--txt)'}} />
            <Line type="monotone" dataKey="IQAIR_AQI" stroke={getCssVar('--c1')} strokeWidth={3} dot={false} />
            <Line type="monotone" dataKey="INMET_PM25" stroke={getCssVar('--c2')} strokeWidth={3} strokeDasharray="5 5" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderChart2 = () => (
    <div className="card">
      <div className="card-title">AQI Diário por Sensor (Março)</div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={dailyBySensorData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartProps.cartesianStroke} vertical={false} />
            <XAxis dataKey="date" stroke={chartProps.axisStroke} fontSize={11} fontWeight={700} />
            <YAxis stroke={chartProps.axisStroke} fontSize={12} />
            <RechartsTooltip contentStyle={TooltipStyle} />
            <Legend wrapperStyle={{fontWeight: 700, fontSize: 11, color: 'var(--txt)'}} />
            <Line type="monotone" dataKey="Brasilia" stroke={getCssVar('--c1')} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="Escola" stroke={getCssVar('--c2')} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="Finatec" stroke={getCssVar('--c3')} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="Gama" stroke={getCssVar('--c4')} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderChart3 = () => (
    <div className="card">
      <div className="card-title">Sazonalidade: PM2.5 Mensal</div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={monthlyInmetData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartProps.cartesianStroke} vertical={false} />
            <XAxis dataKey="month" stroke={chartProps.axisStroke} fontSize={12} fontWeight={700} />
            <YAxis stroke={chartProps.axisStroke} fontSize={12} />
            <RechartsTooltip contentStyle={TooltipStyle} />
            <Legend wrapperStyle={{fontWeight: 700, fontSize: 12, color: 'var(--txt)'}} />
            <Bar dataKey="Media" fill={getCssVar('--c1')} />
            <Bar dataKey="Pico" fill={getCssVar('--c2')} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderChart4 = () => (
    <div className="card">
      <div className="card-title">Distribuição (Histograma PM2.5)</div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={histogramData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartProps.cartesianStroke} vertical={false} />
            <XAxis dataKey="bin" stroke={chartProps.axisStroke} fontSize={12} fontWeight={700} />
            <YAxis stroke={chartProps.axisStroke} fontSize={12} />
            <RechartsTooltip contentStyle={TooltipStyle} />
            <Legend wrapperStyle={{fontWeight: 700, fontSize: 12, color: 'var(--txt)'}} />
            <Area type="monotone" dataKey="Chuva" stackId="1" stroke={getCssVar('--c1')} fill={getCssVar('--c1')} />
            <Area type="monotone" dataKey="Transicao" stackId="1" stroke={getCssVar('--c4')} fill={getCssVar('--c4')} />
            <Area type="monotone" dataKey="Seca" stackId="1" stroke={getCssVar('--c2')} fill={getCssVar('--c2')} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderChartRainEffect = () => (
    <div className="card">
      <div className="card-title">Efeito da Precipitação</div>
      <div className="chart-box" style={{height: 180}}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rainEffectData} layout="vertical" margin={{ left: 40 }}>
            <XAxis type="number" stroke={chartProps.axisStroke} fontSize={12} fontWeight={700} />
            <YAxis dataKey="label" type="category" stroke={chartProps.axisStroke} fontSize={11} fontWeight={700} />
            <RechartsTooltip contentStyle={TooltipStyle} />
            <Bar dataKey="PM2.5" fill={getCssVar('--c3')} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderModelStats = () => (
    <div className="card" style={{ border: `1px solid ${getCssVar('--c1')}` }}>
      <div style={{ fontWeight: 800, color: getCssVar('--c1'), marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8, fontSize: 18 }}>
        <BrainCircuit size={24} strokeWidth={2.5} /> Modelo XGBoost
      </div>
      <div className="grid grid-cols-2">
        <div>
          <div style={{ fontSize: 13, color: 'var(--txt2)', fontWeight: 800 }}>Erro Absoluto (MAE)</div>
          <div style={{ fontSize: 32, fontWeight: 900, color: getCssVar('--c2') }}>{data.model.mae}</div>
        </div>
        <div>
          <div style={{ fontSize: 13, color: 'var(--txt2)', fontWeight: 800 }}>Score (R²)</div>
          <div style={{ fontSize: 32, fontWeight: 900, color: getCssVar('--c1') }}>{data.model.r2}</div>
        </div>
      </div>
    </div>
  );

  const renderChartModel = () => (
    <div className="card">
      <div className="card-title">Previsão vs Real (Teste)</div>
      <div className="chart-box" style={{ height: 250 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={modelData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartProps.cartesianStroke} vertical={false} />
            <XAxis dataKey="id" stroke={chartProps.axisStroke} fontSize={12} tick={false} />
            <YAxis stroke={chartProps.axisStroke} fontSize={12} />
            <RechartsTooltip contentStyle={TooltipStyle} />
            <Legend wrapperStyle={{fontWeight: 700, fontSize: 12, color: 'var(--txt)'}} />
            <Line type="step" name="Real" dataKey="Real" stroke={getCssVar('--c2')} strokeWidth={2.5} dot={false} />
            <Line type="step" name="Previsto" dataKey="Previsto" stroke={getCssVar('--c1')} strokeWidth={2.5} strokeDasharray="4 4" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderChartScatter = () => (
    <div className="card">
      <div className="card-title">Correlação: PM2.5 x NO2</div>
      <div className="chart-box" style={{ height: 250 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke={chartProps.cartesianStroke} />
            <XAxis type="number" dataKey="no2" name="NO2" unit="µg" stroke={chartProps.axisStroke} fontWeight={700} fontSize={12} />
            <YAxis type="number" dataKey="pm25" name="PM2.5" unit="µg" stroke={chartProps.axisStroke} fontWeight={700} fontSize={12} />
            <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={TooltipStyle} />
            <Scatter name="Poluentes" data={scatterData} fill={getCssVar('--c1')} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  // --- DESKTOP COMPONENTS ---
  const renderDashboardCards = () => (
    <div className="grid grid-cols-4">
      <div className="card">
        <div className="card-title"><Activity size={18} /> Registros Totais</div>
        <div className="stat-value">{(data.iqair.total + data.inmet.total).toLocaleString('pt-BR')}</div>
        <div className="stat-label">IQAir + INMET</div>
      </div>
      <div className="card">
        <div className="card-title"><Wind size={18} /> AQI Médio</div>
        <div className="stat-value">{data.iqair.aqi_mean.toFixed(1)}</div>
        <div className="stat-label">Máxima de {data.iqair.aqi_max}</div>
      </div>
      <div className="card">
        <div className="card-title"><Thermometer size={18} /> Temp. Média</div>
        <div className="stat-value">{data.iqair.temp_mean.toFixed(1)}°C</div>
        <div className="stat-label">Umidade: {data.iqair.rh_mean.toFixed(0)}%</div>
      </div>
      <div className="card">
        <div className="card-title"><MapPin size={18} /> Sensores</div>
        <div className="stat-value">{data.iqair.sensors.length}</div>
        <div className="stat-label">Ativos em Brasília/DF</div>
      </div>
    </div>
  );

  // --- MOBILE (APP-LIKE) COMPONENTS ---
  const renderMobileHome = () => (
    <div style={{ padding: '0 20px' }}>
      <div className="card" style={{ borderLeft: `8px solid ${isFavorable ? getCssVar('--c3') : getCssVar('--c2')}`, padding: 16 }}>
        <div style={{ fontSize: 13, color: 'var(--txt2)', fontWeight: 800, textTransform: 'uppercase' }}>Índice de Saída Atual</div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginTop: 8 }}>
          <div className="stat-value" style={{color: isFavorable ? getCssVar('--c3') : getCssVar('--c2')}}>
            {isFavorable ? '82' : '45'} <span style={{fontSize: 20, color: 'var(--txt3)'}}>/100</span>
          </div>
          <div style={{ background: isFavorable ? getCssVar('--c3') : getCssVar('--c2'), color: '#FFF', padding: '6px 14px', borderRadius: 100, fontSize: 13, fontWeight: 800 }}>
            {isFavorable ? 'FAVORÁVEL' : 'ATENÇÃO'}
          </div>
        </div>
        <div style={{ marginTop: 12, fontSize: 14, color: 'var(--txt2)', lineHeight: 1.5, fontWeight: 600 }}>
          {isFavorable ? 'Boas condições. Capacete com viseira padrão é suficiente.' : 'AQI elevado. Recomendado uso de filtro no capacete.'}
        </div>
      </div>

      <div className="scroll-row" style={{ marginTop: 16, padding: 0 }}>
        <div className="pill" style={{minWidth: 100}}>
          <Wind size={22} color={getCssVar('--c1')} />
          <div><div className="pill-val">{data.iqair.aqi_mean.toFixed(1)}</div><div className="pill-lbl">AQI</div></div>
        </div>
        <div className="pill" style={{minWidth: 100}}>
          <Thermometer size={22} color={getCssVar('--c2')} />
          <div><div className="pill-val">{data.iqair.temp_mean.toFixed(0)}°</div><div className="pill-lbl">Temp</div></div>
        </div>
        <div className="pill" style={{minWidth: 100}}>
          <Droplets size={22} color={getCssVar('--c3')} />
          <div><div className="pill-val">{data.iqair.rh_mean.toFixed(0)}%</div><div className="pill-lbl">Umid</div></div>
        </div>
      </div>

      <div className="section-title">Recomendação de Equipamento</div>
      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12}}>
        <div className="card" style={{padding: 16, textAlign: 'center'}}>
          <div style={{fontSize: 32, marginBottom: 8}}>🪖</div>
          <div style={{fontSize: 14, fontWeight: 800, color: 'var(--txt)'}}>Capacete</div>
          <div style={{fontSize: 12, color: 'var(--txt2)', marginTop: 4}}>Viseira Fechada</div>
        </div>
        <div className="card" style={{padding: 16, textAlign: 'center', borderColor: temp > 22 ? getCssVar('--c2') : 'var(--border)'}}>
          <div style={{fontSize: 32, marginBottom: 8}}>🧥</div>
          <div style={{fontSize: 14, fontWeight: 800, color: 'var(--txt)'}}>Jaqueta</div>
          <div style={{fontSize: 12, color: 'var(--txt2)', marginTop: 4}}>{temp > 22 ? 'Leve/Ventilada' : 'Pesada'}</div>
        </div>
        <div className="card" style={{padding: 16, textAlign: 'center'}}>
          <div style={{fontSize: 32, marginBottom: 8}}>💧</div>
          <div style={{fontSize: 14, fontWeight: 800, color: 'var(--txt)'}}>Hidratação</div>
          <div style={{fontSize: 12, color: 'var(--txt2)', marginTop: 4}}>Normal (500ml)</div>
        </div>
        <div className="card" style={{padding: 16, textAlign: 'center'}}>
          <div style={{fontSize: 32, marginBottom: 8}}>🧤</div>
          <div style={{fontSize: 14, fontWeight: 800, color: 'var(--txt)'}}>Luvas</div>
          <div style={{fontSize: 12, color: 'var(--txt2)', marginTop: 4}}>Verão/Leve</div>
        </div>
      </div>

      <div style={{marginTop: 24}}>
        <div className="section-title">Análise de Dados</div>
        <div style={{marginTop: 16}}></div>
        {renderChart1()}
        <div style={{marginTop: 16}}></div>
        {renderChart2()}
      </div>
    </div>
  );

  const renderMobileForecast = () => (
    <div style={{ padding: '0 20px' }}>
      <div className="card" style={{ background: 'rgba(212, 57, 0, 0.1)', borderColor: getCssVar('--c2'), padding: 16 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
          <ShieldAlert size={28} color={getCssVar('--c2')} />
          <div>
            <div style={{ fontWeight: 800, color: getCssVar('--c2'), marginBottom: 4, fontSize: 16 }}>Temporada de Queimadas</div>
            <div style={{ fontSize: 13, color: 'var(--txt2)', lineHeight: 1.5, fontWeight: 600 }}>
              Jul–Out: PM2.5 sobe drasticamente no Cerrado. Filtro N95 é obrigatório em percursos longos.
            </div>
          </div>
        </div>
      </div>

      <div className="section-title">Melhor Janela para Rodar Hoje</div>
      <div style={{display: 'flex', gap: 12, marginTop: 12}}>
        <div className="card" style={{flex: 1, textAlign: 'center', borderColor: getCssVar('--c3'), background: 'rgba(0, 102, 34, 0.05)'}}>
          <div style={{fontSize: 24, marginBottom: 4}}>🌅</div>
          <div style={{fontSize: 18, fontWeight: 900, color: getCssVar('--c3')}}>11h–14h</div>
          <div style={{fontSize: 12, color: 'var(--txt2)', fontWeight: 600}}>Menor Poluição</div>
        </div>
        <div className="card" style={{flex: 1, textAlign: 'center', borderColor: getCssVar('--c2'), background: 'rgba(212, 57, 0, 0.05)'}}>
          <div style={{fontSize: 24, marginBottom: 4}}>🌆</div>
          <div style={{fontSize: 18, fontWeight: 900, color: getCssVar('--c2')}}>19h–22h</div>
          <div style={{fontSize: 12, color: 'var(--txt2)', fontWeight: 600}}>Inversão Térmica</div>
        </div>
      </div>

      <div className="section-title">Guia Sazonal Rápido (Brasília)</div>
      <div className="card" style={{padding: 0, overflow: 'hidden'}}>
        <div style={{padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div><div style={{fontWeight: 800, fontSize: 15, color: 'var(--txt)'}}>Jan — Jun</div><div style={{fontSize: 12, color: 'var(--txt2)'}}>Chuvoso · PM2.5 Baixo</div></div>
          <div style={{background: getCssVar('--c3'), color: '#fff', padding: '4px 10px', borderRadius: 100, fontSize: 11, fontWeight: 800}}>ÓTIMO</div>
        </div>
        <div style={{padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div><div style={{fontWeight: 800, fontSize: 15, color: 'var(--txt)'}}>Jul — Out</div><div style={{fontSize: 12, color: 'var(--txt2)'}}>Seca/Fumaça · PM2.5 Alto</div></div>
          <div style={{background: getCssVar('--c2'), color: '#fff', padding: '4px 10px', borderRadius: 100, fontSize: 11, fontWeight: 800}}>PERIGO</div>
        </div>
        <div style={{padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div><div style={{fontWeight: 800, fontSize: 15, color: 'var(--txt)'}}>Nov — Dez</div><div style={{fontSize: 12, color: 'var(--txt2)'}}>Transição · PM2.5 Médio</div></div>
          <div style={{background: 'var(--txt3)', color: '#fff', padding: '4px 10px', borderRadius: 100, fontSize: 11, fontWeight: 800}}>MODERADO</div>
        </div>
      </div>

      <div style={{marginTop: 24}}>
        <div className="section-title">Análise Climática</div>
        <div style={{marginTop: 16}}></div>
        {renderChart3()}
        <div style={{marginTop: 16}}></div>
        {renderChart4()}
        <div style={{marginTop: 16}}></div>
        {renderChartRainEffect()}
      </div>
    </div>
  );

  const renderMobileProfile = () => (
    <div style={{ padding: '0 20px' }}>
      <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 16, padding: 20 }}>
        <div style={{ width: 60, height: 60, background: getCssVar('--c1'), borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24, color: '#fff' }}>
          <User />
        </div>
        <div>
          <div style={{ fontSize: 20, fontWeight: 900, color: 'var(--txt)' }}>Motociclista</div>
          <div style={{ fontSize: 13, color: 'var(--txt2)', fontWeight: 600 }}>Brasília, DF</div>
          <div style={{ background: 'rgba(0,0,0,0.1)', display: 'inline-block', padding: '4px 10px', borderRadius: 100, fontSize: 11, fontWeight: 800, marginTop: 6, color: 'var(--txt)' }}>BETA TESTER</div>
        </div>
      </div>

      <div className="section-title">Alertas Inteligentes Ativos</div>
      <div className="card" style={{padding: 0, overflow: 'hidden'}}>
        <div style={{padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div><div style={{fontWeight: 800, fontSize: 14, color: 'var(--txt)'}}>Filtro de Ar (Capacete)</div><div style={{fontSize: 12, color: 'var(--txt2)'}}>Alerta quando PM2.5 &gt; 12 µg/m³</div></div>
          <div style={{width: 44, height: 24, background: getCssVar('--c1'), borderRadius: 100, position: 'relative'}}><div style={{position: 'absolute', right: 2, top: 2, width: 20, height: 20, background: '#fff', borderRadius: '50%'}}></div></div>
        </div>
        <div style={{padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div><div style={{fontWeight: 800, fontSize: 14, color: 'var(--txt)'}}>Previsão de Chuva</div><div style={{fontSize: 12, color: 'var(--txt2)'}}>Avisar 30min antes de rodar</div></div>
          <div style={{width: 44, height: 24, background: getCssVar('--c1'), borderRadius: 100, position: 'relative'}}><div style={{position: 'absolute', right: 2, top: 2, width: 20, height: 20, background: '#fff', borderRadius: '50%'}}></div></div>
        </div>
      </div>

      <div className="section-title">Fontes de Dados</div>
      <div className="card" style={{padding: 16}}>
        <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}><span style={{fontWeight: 700}}>IQAir API</span><span style={{color: getCssVar('--c3'), fontWeight: 800, fontSize: 12}}>CONECTADO</span></div>
        <div style={{display: 'flex', justifyContent: 'space-between'}}><span style={{fontWeight: 700}}>INMET Brasil</span><span style={{color: getCssVar('--c3'), fontWeight: 800, fontSize: 12}}>CONECTADO</span></div>
        <div style={{marginTop: 16, fontSize: 12, color: 'var(--txt2)', lineHeight: 1.4}}>O app usa {data.iqair.total.toLocaleString()} registros reais para sugerir o melhor uso do equipamento.</div>
      </div>

      <div style={{marginTop: 24}}>
        <div className="section-title">Métricas do Modelo XGBoost</div>
        <div style={{marginTop: 16}}></div>
        {renderModelStats()}
        <div style={{marginTop: 16}}></div>
        {renderChartModel()}
        <div style={{marginTop: 16}}></div>
        {renderChartScatter()}
      </div>
    </div>
  );

  return (
    <>
      <div className="top-bar">
        <div className="view-toggles">
          <button className={`toggle-btn ${viewMode === 'desktop' ? 'active' : ''}`} onClick={() => {setViewMode('desktop'); setActiveTab('home');}}>
            <Monitor size={18} /> Desktop (Dados)
          </button>
          <button className={`toggle-btn ${viewMode === 'mobile' ? 'active' : ''}`} onClick={() => {setViewMode('mobile'); setActiveTab('home');}}>
            <Smartphone size={18} /> Mobile (App)
          </button>
        </div>
        <div className="view-toggles">
          <button className={`toggle-btn`} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} style={{ marginRight: 16 }}>
            {theme === 'dark' ? <SunMoon size={18}/> : <Moon size={18}/>} 
            {theme === 'dark' ? 'Modo Claro' : 'Modo Escuro'}
          </button>
          <button className="export-btn" onClick={exportPDF} disabled={isExporting}>
            <FileDown size={18} />
            {isExporting ? 'Gerando...' : 'Exportar PDF'}
          </button>
        </div>
      </div>

      <div>
        {viewMode === 'desktop' ? (
          <div className="desktop-layout">
            <aside className="sidebar">
              <div style={{ fontSize: 28, fontWeight: 900, marginBottom: 32, padding: 8 }}>
                MOTO<span style={{ color: getCssVar('--c1') }}>AR</span>
              </div>
              <div className={`nav-item-desktop ${activeTab === 'home' ? 'active' : ''}`} onClick={() => setActiveTab('home')}>
                <Activity size={20} /> Geral
              </div>
              <div className={`nav-item-desktop ${activeTab === 'climate' ? 'active' : ''}`} onClick={() => setActiveTab('climate')}>
                <CloudRain size={20} /> Clima
              </div>
              <div className={`nav-item-desktop ${activeTab === 'model' ? 'active' : ''}`} onClick={() => setActiveTab('model')}>
                <BrainCircuit size={20} /> Predição
              </div>
            </aside>
            <main className="main-content">
              {activeTab === 'home' && (
                <>
                  <div style={{ fontSize: 24, fontWeight: 800, marginBottom: 24 }}>Visão Geral</div>
                  {renderDashboardCards()}
                  <div className="grid grid-cols-2">
                    {renderChart1()}
                    {renderChart2()}
                  </div>
                  <div className="grid grid-cols-2">
                    <div className="card">
                      <div className="card-title">AQI por Sensor (Boxplot)</div>
                      <div className="chart-box">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={boxPlotData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={chartProps.cartesianStroke} vertical={false} />
                            <XAxis dataKey="sensor" stroke={chartProps.axisStroke} fontSize={11} fontWeight={700} />
                            <YAxis stroke={chartProps.axisStroke} fontSize={12} />
                            <RechartsTooltip contentStyle={TooltipStyle} />
                            <Legend wrapperStyle={{fontWeight: 700, fontSize: 11, color: 'var(--txt)'}} />
                            <Bar dataKey="q1" stackId="box" fill="transparent" />
                            <Bar dataKey="median" fill={getCssVar('--c1')} name="Mediana" />
                            <Bar dataKey="q3" fill={getCssVar('--c2')} name="Q3" />
                            <Bar dataKey="max" fill={getCssVar('--c4')} name="Máx" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                    <div className="card">
                      <div className="card-title">Sensores IQAir — Estatísticas</div>
                      <div style={{overflowX: 'auto'}}>
                        <table className="data-table">
                          <thead>
                            <tr><th>Sensor</th><th>Registros</th><th>AQI Médio</th><th>AQI Máx</th><th>Desvio</th><th>PM2.5</th></tr>
                          </thead>
                          <tbody>
                            {sensorStatsData.map((s: any) => (
                              <tr key={s.sensor_location}>
                                <td style={{fontWeight: 800, color: 'var(--txt)'}}>{s.sensor_location}</td>
                                <td>{s.records.toLocaleString()}</td>
                                <td>{s.aqi_mean}</td>
                                <td>{s.aqi_max}</td>
                                <td>{s.aqi_std}</td>
                                <td>{s.pm25_mean}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                  <div className="card" style={{marginTop: 24}}>
                    <div className="card-title">Estatísticas Descritivas (INMET)</div>
                    <div style={{overflowX: 'auto'}}>
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Variável</th>
                            {data.inmet.stats_table.cols.slice(1).map((c: string) => <th key={c}>{c}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {data.inmet.stats_table.rows.map((r: string, i: number) => (
                            <tr key={r}>
                              <td style={{fontWeight: 800, color: 'var(--txt)'}}>{r}</td>
                              {data.inmet.stats_table.data[i].slice(1).map((val: number, j: number) => (
                                <td key={j}>{val.toFixed(2)}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
              {activeTab === 'climate' && (
                <>
                  <div style={{ fontSize: 24, fontWeight: 800, marginBottom: 24 }}>Análise Climática e Sazonalidade</div>
                  <div className="grid grid-cols-2">
                    {renderChart3()}
                    {renderChart4()}
                  </div>
                  <div className="grid grid-cols-2">
                    <div className="card">
                      <div className="card-title">Perfil Horário por Estação</div>
                      <div className="chart-box">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={hourlyBySeasonData2}>
                            <CartesianGrid strokeDasharray="3 3" stroke={chartProps.cartesianStroke} vertical={false} />
                            <XAxis dataKey="hour" stroke={chartProps.axisStroke} fontSize={12} fontWeight={700} />
                            <YAxis stroke={chartProps.axisStroke} fontSize={12} />
                            <RechartsTooltip contentStyle={TooltipStyle} />
                            <Legend wrapperStyle={{fontWeight: 700, fontSize: 12, color: 'var(--txt)'}} />
                            <Line type="monotone" dataKey="Chuva" stroke={getCssVar('--c1')} strokeWidth={3} dot={false} />
                            <Line type="monotone" dataKey="Seca" stroke={getCssVar('--c2')} strokeWidth={3} dot={false} />
                            <Line type="monotone" dataKey="Transicao" stroke={getCssVar('--c3')} strokeWidth={3} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                    <div className="card">
                      <div className="card-title">% Horas Acima da OMS (15 µg/m³)</div>
                      <div className="chart-box">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={omsData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={chartProps.cartesianStroke} vertical={false} />
                            <XAxis dataKey="month" stroke={chartProps.axisStroke} fontSize={12} fontWeight={700} />
                            <YAxis stroke={chartProps.axisStroke} fontSize={12} unit="%" />
                            <RechartsTooltip contentStyle={TooltipStyle} formatter={(v) => `${(v as number).toFixed(1)}%`} />
                            <Bar dataKey="pct" fill={getCssVar('--c2')} name="% > OMS" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                  <div className="grid grid-cols-2">
                    {renderChartRainEffect()}
                    <div className="card">
                      <div className="card-title">Matriz de Correlação (Poluentes)</div>
                      <div style={{overflowX: 'auto'}}>
                        <table className="data-table">
                          <thead>
                            <tr><th></th>{corrLabels.map((l: string) => <th key={l}>{l}</th>)}</tr>
                          </thead>
                          <tbody>
                            {corrLabels.map((row: string, i: number) => (
                              <tr key={row}>
                                <td style={{fontWeight: 800}}>{row}</td>
                                {corrMatrix[i].map((v: number, j: number) => (
                                  <td key={j} style={{background: v > 0.3 ? 'rgba(0,85,255,0.15)' : v < -0.05 ? 'rgba(212,57,0,0.15)' : 'transparent', fontWeight: Math.abs(v) > 0.3 ? 800 : 400}}>
                                    {v.toFixed(3)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </>
              )}
              {activeTab === 'model' && (
                <>
                  <div style={{ fontSize: 24, fontWeight: 800, marginBottom: 24 }}>Performance do Modelo Preditivo</div>
                  {renderModelStats()}
                  <div style={{ marginTop: 24 }} className="grid grid-cols-2">
                    {renderChartModel()}
                    {renderChartScatter()}
                  </div>
                  <div className="grid grid-cols-2" style={{marginTop: 24}}>
                    <div className="card">
                      <div className="card-title">Importância das Features (%)</div>
                      <div className="chart-box" style={{height: 300}}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={featureImportanceData} layout="vertical" margin={{ left: 60 }}>
                            <XAxis type="number" stroke={chartProps.axisStroke} fontSize={12} unit="%" />
                            <YAxis dataKey="name" type="category" stroke={chartProps.axisStroke} fontSize={11} fontWeight={700} />
                            <RechartsTooltip contentStyle={TooltipStyle} formatter={(v) => `${(v as number).toFixed(2)}%`} />
                            <Bar dataKey="importance" fill={getCssVar('--c1')} name="Importância" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                    <div className="card">
                      <div className="card-title">Metadados do Modelo</div>
                      <div style={{padding: 8}}>
                        <table className="data-table">
                          <tbody>
                            <tr><td style={{fontWeight: 800}}>Algoritmo</td><td>{data.model.name}</td></tr>
                            <tr><td style={{fontWeight: 800}}>Amostras Treino</td><td>{data.model.n_train.toLocaleString()}</td></tr>
                            <tr><td style={{fontWeight: 800}}>Amostras Teste</td><td>{data.model.n_test.toLocaleString()}</td></tr>
                            <tr><td style={{fontWeight: 800}}>MAE</td><td>{data.model.mae}</td></tr>
                            <tr><td style={{fontWeight: 800}}>R²</td><td>{data.model.r2}</td></tr>
                            <tr><td style={{fontWeight: 800}}>Features</td><td>{data.model.features.length}</td></tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </main>
          </div>
        ) : (
          <div className="mobile-wrapper">
            <div className="phone-frame">
              <div className="notch"></div>
              <div className="mobile-status-bar">
                <span>9:41</span>
                <span>100%</span>
              </div>
              <div className="mobile-screen">
                <div className="mobile-header">
                  <div className="mobile-app-name">MOTO<span>AR</span></div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--txt2)' }}>Para Motociclistas</div>
                </div>

                {activeTab === 'home' && renderMobileHome()}
                {activeTab === 'climate' && renderMobileForecast()}
                {activeTab === 'model' && renderMobileProfile()}
              </div>
              <div className="bottom-nav">
                <div className={`nav-item-mobile ${activeTab === 'home' ? 'active' : ''}`} onClick={() => setActiveTab('home')}>
                  <Activity size={24} strokeWidth={activeTab==='home'?3:2} />
                  <span style={{fontSize: 10, fontWeight: 800, marginTop: 4}}>AGORA</span>
                </div>
                <div className={`nav-item-mobile ${activeTab === 'climate' ? 'active' : ''}`} onClick={() => setActiveTab('climate')}>
                  <CalendarDays size={24} strokeWidth={activeTab==='climate'?3:2} />
                  <span style={{fontSize: 10, fontWeight: 800, marginTop: 4}}>PREVISÃO</span>
                </div>
                <div className={`nav-item-mobile ${activeTab === 'model' ? 'active' : ''}`} onClick={() => setActiveTab('model')}>
                  <User size={24} strokeWidth={activeTab==='model'?3:2} />
                  <span style={{fontSize: 10, fontWeight: 800, marginTop: 4}}>PERFIL</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* RELATÓRIO PDF OCULTO */}
      {showReport && (
        <div ref={reportRef} style={{
          position: 'fixed', top: 0, left: '-9999px', width: '794px',
          background: '#fff', color: '#000', fontFamily: 'Inter, sans-serif', fontSize: 13, lineHeight: 1.6,
          padding: '40px 50px'
        }}>
          {/* CAPA */}
          <div style={{textAlign: 'center', marginBottom: 50, paddingTop: 60}}>
            <div style={{fontSize: 42, fontWeight: 900, color: '#111', letterSpacing: -1}}>MOTO<span style={{color: '#0055FF'}}>AR</span></div>
            <div style={{fontSize: 16, color: '#666', marginTop: 8, fontWeight: 600}}>Relatório Técnico de Qualidade do Ar</div>
            <div style={{width: 80, height: 4, background: '#0055FF', margin: '20px auto', borderRadius: 2}}></div>
            <div style={{fontSize: 13, color: '#888', marginTop: 16}}>Brasília/DF — {new Date().toLocaleDateString('pt-BR', {year:'numeric',month:'long',day:'numeric'})}</div>
            <div style={{fontSize: 12, color: '#aaa', marginTop: 8}}>Fontes: IQAir ({data.iqair.total.toLocaleString()} registros) + INMET ({data.inmet.total.toLocaleString()} registros)</div>
          </div>

          {/* RESUMO EXECUTIVO */}
          <div style={{pageBreakBefore: 'always'}}></div>
          <h2 style={{fontSize: 20, fontWeight: 800, borderBottom: '3px solid #0055FF', paddingBottom: 6, marginBottom: 20}}>1. Resumo Executivo</h2>
          <div style={{display:'grid', gridTemplateColumns:'1fr 1fr 1fr 1fr', gap: 16, marginBottom: 30}}>
            <div style={{border:'1px solid #ddd', borderRadius:8, padding:16, textAlign:'center'}}>
              <div style={{fontSize:11, color:'#888', fontWeight:700}}>REGISTROS</div>
              <div style={{fontSize:28, fontWeight:900}}>{(data.iqair.total + data.inmet.total).toLocaleString('pt-BR')}</div>
            </div>
            <div style={{border:'1px solid #ddd', borderRadius:8, padding:16, textAlign:'center'}}>
              <div style={{fontSize:11, color:'#888', fontWeight:700}}>AQI MÉDIO</div>
              <div style={{fontSize:28, fontWeight:900}}>{data.iqair.aqi_mean.toFixed(1)}</div>
            </div>
            <div style={{border:'1px solid #ddd', borderRadius:8, padding:16, textAlign:'center'}}>
              <div style={{fontSize:11, color:'#888', fontWeight:700}}>TEMP. MÉDIA</div>
              <div style={{fontSize:28, fontWeight:900}}>{data.iqair.temp_mean.toFixed(1)}°C</div>
            </div>
            <div style={{border:'1px solid #ddd', borderRadius:8, padding:16, textAlign:'center'}}>
              <div style={{fontSize:11, color:'#888', fontWeight:700}}>SENSORES</div>
              <div style={{fontSize:28, fontWeight:900}}>{data.iqair.sensors.length}</div>
            </div>
          </div>

          {/* GRÁFICO 1 */}
          <h2 style={{fontSize: 20, fontWeight: 800, borderBottom: '3px solid #0055FF', paddingBottom: 6, marginBottom: 20}}>2. Comparação IQAir vs INMET</h2>
          <div style={{height: 280, marginBottom: 30}}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={hourlyCompareData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ddd" vertical={false} />
                <XAxis dataKey="hour" stroke="#666" fontSize={11} />
                <YAxis stroke="#666" fontSize={11} />
                <Legend wrapperStyle={{fontSize: 11}} />
                <Line type="monotone" dataKey="IQAIR_AQI" stroke="#0055FF" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="INMET_PM25" stroke="#D43900" strokeWidth={2} strokeDasharray="5 5" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* TABELA SENSORES */}
          <h3 style={{fontSize:16, fontWeight:800, marginBottom:10}}>2.1 Estatísticas por Sensor (IQAir)</h3>
          <table style={{width:'100%', borderCollapse:'collapse', marginBottom:30, fontSize:12}}>
            <thead>
              <tr style={{background:'#f5f5f5'}}>
                <th style={{padding:10, textAlign:'left', borderBottom:'2px solid #000'}}>Sensor</th>
                <th style={{padding:10, textAlign:'right', borderBottom:'2px solid #000'}}>Registros</th>
                <th style={{padding:10, textAlign:'right', borderBottom:'2px solid #000'}}>AQI Médio</th>
                <th style={{padding:10, textAlign:'right', borderBottom:'2px solid #000'}}>AQI Máx</th>
                <th style={{padding:10, textAlign:'right', borderBottom:'2px solid #000'}}>σ</th>
                <th style={{padding:10, textAlign:'right', borderBottom:'2px solid #000'}}>PM2.5</th>
              </tr>
            </thead>
            <tbody>
              {sensorStatsData.map((s: any) => (
                <tr key={s.sensor_location}>
                  <td style={{padding:10, borderBottom:'1px solid #eee', fontWeight:700}}>{s.sensor_location}</td>
                  <td style={{padding:10, borderBottom:'1px solid #eee', textAlign:'right'}}>{s.records.toLocaleString()}</td>
                  <td style={{padding:10, borderBottom:'1px solid #eee', textAlign:'right'}}>{s.aqi_mean}</td>
                  <td style={{padding:10, borderBottom:'1px solid #eee', textAlign:'right'}}>{s.aqi_max}</td>
                  <td style={{padding:10, borderBottom:'1px solid #eee', textAlign:'right'}}>{s.aqi_std}</td>
                  <td style={{padding:10, borderBottom:'1px solid #eee', textAlign:'right'}}>{s.pm25_mean}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* SAZONALIDADE */}
          <div style={{pageBreakBefore: 'always'}}></div>
          <h2 style={{fontSize: 20, fontWeight: 800, borderBottom: '3px solid #0055FF', paddingBottom: 6, marginBottom: 20}}>3. Análise Climática</h2>
          <h3 style={{fontSize:16, fontWeight:800, marginBottom:10}}>3.1 PM2.5 Mensal (Média e Percentil 75)</h3>
          <div style={{height: 260, marginBottom: 30}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={monthlyInmetData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ddd" vertical={false} />
                <XAxis dataKey="month" stroke="#666" fontSize={11} />
                <YAxis stroke="#666" fontSize={11} />
                <Legend wrapperStyle={{fontSize: 11}} />
                <Bar dataKey="Media" fill="#0055FF" />
                <Bar dataKey="Pico" fill="#D43900" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <h3 style={{fontSize:16, fontWeight:800, marginBottom:10}}>3.2 Perfil Horário por Estação</h3>
          <div style={{height: 260, marginBottom: 30}}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={hourlyBySeasonData2}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ddd" vertical={false} />
                <XAxis dataKey="hour" stroke="#666" fontSize={11} />
                <YAxis stroke="#666" fontSize={11} />
                <Legend wrapperStyle={{fontSize: 11}} />
                <Line type="monotone" dataKey="Chuva" stroke="#0055FF" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Seca" stroke="#D43900" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Transicao" stroke="#006622" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <h3 style={{fontSize:16, fontWeight:800, marginBottom:10}}>3.3 % Horas Acima do Limite OMS (15 µg/m³)</h3>
          <div style={{height: 240, marginBottom: 30}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={omsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ddd" vertical={false} />
                <XAxis dataKey="month" stroke="#666" fontSize={11} />
                <YAxis stroke="#666" fontSize={11} unit="%" />
                <Bar dataKey="pct" fill="#D43900" name="% > OMS" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <h3 style={{fontSize:16, fontWeight:800, marginBottom:10}}>3.4 Correlação entre Poluentes</h3>
          <table style={{width:'100%', borderCollapse:'collapse', marginBottom:30, fontSize:12}}>
            <thead>
              <tr style={{background:'#f5f5f5'}}>
                <th style={{padding:8, borderBottom:'2px solid #000'}}></th>
                {corrLabels.map((l: string) => <th key={l} style={{padding:8, borderBottom:'2px solid #000', textAlign:'right'}}>{l}</th>)}
              </tr>
            </thead>
            <tbody>
              {corrLabels.map((row: string, i: number) => (
                <tr key={row}>
                  <td style={{padding:8, fontWeight:700, borderBottom:'1px solid #eee'}}>{row}</td>
                  {corrMatrix[i].map((v: number, j: number) => (
                    <td key={j} style={{padding:8, textAlign:'right', borderBottom:'1px solid #eee', fontWeight: Math.abs(v) > 0.3 ? 800 : 400, background: v > 0.3 ? '#e8f0fe' : v < -0.05 ? '#fde8e8' : 'transparent'}}>{v.toFixed(3)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          {/* MODELO */}
          <div style={{pageBreakBefore: 'always'}}></div>
          <h2 style={{fontSize: 20, fontWeight: 800, borderBottom: '3px solid #0055FF', paddingBottom: 6, marginBottom: 20}}>4. Modelo Preditivo ({data.model.name})</h2>
          <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap: 24, marginBottom: 30}}>
            <div style={{border:'1px solid #ddd', borderRadius:8, padding:20}}>
              <div style={{fontSize:12, color:'#888', fontWeight:700}}>ERRO ABSOLUTO MÉDIO (MAE)</div>
              <div style={{fontSize:36, fontWeight:900, color:'#D43900'}}>{data.model.mae}</div>
            </div>
            <div style={{border:'1px solid #ddd', borderRadius:8, padding:20}}>
              <div style={{fontSize:12, color:'#888', fontWeight:700}}>COEFICIENTE R²</div>
              <div style={{fontSize:36, fontWeight:900, color:'#0055FF'}}>{data.model.r2}</div>
            </div>
          </div>
          <div style={{display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap: 12, marginBottom: 30, fontSize: 12}}>
            <div style={{border:'1px solid #eee', borderRadius:8, padding:12, textAlign:'center'}}><strong>Algoritmo:</strong> {data.model.name}</div>
            <div style={{border:'1px solid #eee', borderRadius:8, padding:12, textAlign:'center'}}><strong>Treino:</strong> {data.model.n_train.toLocaleString()} amostras</div>
            <div style={{border:'1px solid #eee', borderRadius:8, padding:12, textAlign:'center'}}><strong>Teste:</strong> {data.model.n_test.toLocaleString()} amostras</div>
          </div>

          <h3 style={{fontSize:16, fontWeight:800, marginBottom:10}}>4.1 Previsão vs Valor Real</h3>
          <div style={{height: 260, marginBottom: 30}}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={modelData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ddd" vertical={false} />
                <XAxis dataKey="id" stroke="#666" fontSize={11} tick={false} />
                <YAxis stroke="#666" fontSize={11} />
                <Legend wrapperStyle={{fontSize: 11}} />
                <Line type="step" name="Real" dataKey="Real" stroke="#D43900" strokeWidth={2} dot={false} />
                <Line type="step" name="Previsto" dataKey="Previsto" stroke="#0055FF" strokeWidth={2} strokeDasharray="4 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <h3 style={{fontSize:16, fontWeight:800, marginBottom:10}}>4.2 Importância das Features</h3>
          <div style={{height: 280, marginBottom: 30}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureImportanceData} layout="vertical" margin={{ left: 80 }}>
                <XAxis type="number" stroke="#666" fontSize={11} unit="%" />
                <YAxis dataKey="name" type="category" stroke="#666" fontSize={11} />
                <Bar dataKey="importance" fill="#0055FF" name="Importância" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <h3 style={{fontSize:16, fontWeight:800, marginBottom:10}}>4.3 Dispersão PM2.5 × NO2</h3>
          <div style={{height: 260, marginBottom: 30}}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#ddd" />
                <XAxis type="number" dataKey="no2" name="NO2" unit="µg" stroke="#666" fontSize={11} />
                <YAxis type="number" dataKey="pm25" name="PM2.5" unit="µg" stroke="#666" fontSize={11} />
                <Scatter name="Poluentes" data={scatterData} fill="#0055FF" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* TABELA DESCRITIVA */}
          <div style={{pageBreakBefore: 'always'}}></div>
          <h2 style={{fontSize: 20, fontWeight: 800, borderBottom: '3px solid #0055FF', paddingBottom: 6, marginBottom: 20}}>5. Dados Descritivos (INMET)</h2>
          <table style={{width:'100%', borderCollapse:'collapse', fontSize:12}}>
            <thead>
              <tr style={{background:'#f5f5f5'}}>
                <th style={{padding:10, textAlign:'left', borderBottom:'2px solid #000'}}>Variável</th>
                {data.inmet.stats_table.cols.slice(1).map((c: string) => <th key={c} style={{padding:10, textAlign:'right', borderBottom:'2px solid #000'}}>{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {data.inmet.stats_table.rows.map((r: string, i: number) => (
                <tr key={r}>
                  <td style={{padding:10, fontWeight:700, borderBottom:'1px solid #eee'}}>{r}</td>
                  {data.inmet.stats_table.data[i].slice(1).map((val: number, j: number) => (
                    <td key={j} style={{padding:10, textAlign:'right', borderBottom:'1px solid #eee'}}>{val.toFixed(2)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          {/* RODAPÉ */}
          <div style={{marginTop: 60, borderTop: '2px solid #ddd', paddingTop: 20, textAlign: 'center', color: '#999', fontSize: 11}}>
            Gerado automaticamente pelo sistema MotoAR — {new Date().toLocaleDateString('pt-BR')} às {new Date().toLocaleTimeString('pt-BR')}
          </div>
        </div>
      )}
    </>
  );
}
