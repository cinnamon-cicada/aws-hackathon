import React, { useState, useEffect, useRef } from 'react';
import { MapPin, Activity, Users, AlertTriangle, Clock, Play, Pause } from 'lucide-react';

// Mock data generator
const generateMockData = () => {
  const cells = [];
  const survivors = [];
  
  // Generate 20x20 grid of cells
  for (let lat = 0; lat < 20; lat++) {
    for (let lon = 0; lon < 20; lon++) {
      const damageLevel = Math.random();
      const hasSurvivors = Math.random() > 0.7;
      
      cells.push({
        id: `${lat}_${lon}`,
        lat,
        lon,
        damageLevel,
        preBuildings: Math.floor(Math.random() * 50) + 10,
        postBuildings: Math.floor((1 - damageLevel) * 50),
        population: Math.floor(Math.random() * 500) + 100,
        hazards: damageLevel > 0.7 ? ['COLLAPSE'] : damageLevel > 0.5 ? ['GAS_LEAK'] : [],
        lastUpdate: Date.now() - Math.random() * 7200000,
        freshnessScore: Math.random()
      });
      
      if (hasSurvivors) {
        const count = Math.floor(Math.random() * 3) + 1;
        for (let i = 0; i < count; i++) {
          survivors.push({
            id: `surv_${lat}_${lon}_${i}`,
            lat: lat + Math.random(),
            lon: lon + Math.random(),
            urgency: Math.random(),
            status: Math.random() > 0.7 ? 'CRITICAL' : Math.random() > 0.4 ? 'MEDIUM' : 'STABLE',
            detectionSources: ['THERMAL', 'RGB', 'SOUND'].slice(0, Math.floor(Math.random() * 3) + 1),
            firstDetected: Date.now() - Math.random() * 86400000,
            lastUpdated: Date.now() - Math.random() * 3600000
          });
        }
      }
    }
  }
  
  return { cells, survivors };
};

const DisasterResponseDemo = () => {
  const [data, setData] = useState(generateMockData());
  const [viewMode, setViewMode] = useState('after'); // 'before', 'after', 'split'
  const [layers, setLayers] = useState({
    damage: true,
    survivors: true,
    hazards: true,
    population: false
  });
  const [selectedSurvivor, setSelectedSurvivor] = useState(null);
  const [hoveredCell, setHoveredCell] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [timeOffset, setTimeOffset] = useState(0);
  const canvasRef = useRef(null);
  
  // Metrics calculation
  const metrics = {
    coveragePercent: Math.floor((data.cells.filter(c => c.freshnessScore > 0.5).length / data.cells.length) * 100),
    survivorsFound: data.survivors.length,
    criticalCount: data.survivors.filter(s => s.status === 'CRITICAL').length,
    avgUrgency: (data.survivors.reduce((sum, s) => sum + s.urgency, 0) / data.survivors.length * 100).toFixed(0)
  };
  
  // Auto-refresh simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => ({
        ...prev,
        cells: prev.cells.map(cell => ({
          ...cell,
          freshnessScore: Math.max(0, cell.freshnessScore - 0.01)
        }))
      }));
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Timeline playback
  useEffect(() => {
    if (!isPlaying) return;
    
    const interval = setInterval(() => {
      setTimeOffset(prev => {
        if (prev >= 120) {
          setIsPlaying(false);
          return 0;
        }
        return prev + 1;
      });
    }, 100);
    
    return () => clearInterval(interval);
  }, [isPlaying]);
  
  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const cellSize = 30;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw cells
    if (layers.damage) {
      data.cells.forEach(cell => {
        const x = cell.lon * cellSize;
        const y = cell.lat * cellSize;
        
        // Damage color
        let color;
        if (viewMode === 'before') {
          color = '#4CAF50';
        } else {
          const damage = cell.damageLevel;
          if (damage < 0.2) color = '#00ff00';
          else if (damage < 0.4) color = '#ffff00';
          else if (damage < 0.6) color = '#ff8800';
          else if (damage < 0.8) color = '#ff0000';
          else color = '#8b0000';
        }
        
        // Apply freshness
        ctx.globalAlpha = viewMode === 'after' ? cell.freshnessScore * 0.8 + 0.2 : 0.8;
        ctx.fillStyle = color;
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
        
        // Highlight hovered cell
        if (hoveredCell?.id === cell.id) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, cellSize - 1, cellSize - 1);
        }
      });
    }
    
    ctx.globalAlpha = 1;
    
    // Draw hazards
    if (layers.hazards && viewMode !== 'before') {
      data.cells.forEach(cell => {
        if (cell.hazards.length > 0) {
          const x = cell.lon * cellSize + cellSize / 2;
          const y = cell.lat * cellSize + cellSize / 2;
          
          ctx.fillStyle = '#ff0000';
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fill();
          
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x, y - 8);
          ctx.lineTo(x, y + 8);
          ctx.stroke();
        }
      });
    }
    
    // Draw survivors
    if (layers.survivors && viewMode !== 'before') {
      data.survivors.forEach(survivor => {
        const x = survivor.lon * cellSize;
        const y = survivor.lat * cellSize;
        
        // Pulsing effect for critical
        const pulse = survivor.status === 'CRITICAL' ? Math.sin(Date.now() / 200) * 0.3 + 0.7 : 1;
        const size = 8 * pulse;
        
        // Status color
        let color;
        if (survivor.status === 'CRITICAL') color = '#ff0000';
        else if (survivor.status === 'MEDIUM') color = '#ff8800';
        else color = '#00ff00';
        
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.stroke();
        
        // Highlight selected
        if (selectedSurvivor?.id === survivor.id) {
          ctx.strokeStyle = '#00ffff';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(x, y, size + 4, 0, Math.PI * 2);
          ctx.stroke();
        }
      });
    }
  }, [data, layers, viewMode, hoveredCell, selectedSurvivor]);
  
  // Canvas click handler
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const cellSize = 30;
    
    // Check for survivor click
    const clickedSurvivor = data.survivors.find(s => {
      const sx = s.lon * cellSize;
      const sy = s.lat * cellSize;
      const dist = Math.sqrt((x - sx) ** 2 + (y - sy) ** 2);
      return dist < 10;
    });
    
    if (clickedSurvivor) {
      setSelectedSurvivor(clickedSurvivor);
    }
  };
  
  const handleCanvasHover = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const cellSize = 30;
    
    const cellLon = Math.floor(x / cellSize);
    const cellLat = Math.floor(y / cellSize);
    const cell = data.cells.find(c => c.lon === cellLon && c.lat === cellLat);
    
    setHoveredCell(cell || null);
  };
  
  const sortedSurvivors = [...data.survivors].sort((a, b) => b.urgency - a.urgency);
  
  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Left Sidebar - Controls */}
      <div className="w-64 bg-gray-800 p-4 overflow-y-auto">
        <h2 className="text-xl font-bold mb-4">Disaster Response</h2>
        
        {/* View Mode */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold mb-2">View Mode</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setViewMode('before')}
              className={`px-3 py-1 rounded text-xs ${viewMode === 'before' ? 'bg-blue-600' : 'bg-gray-700'}`}
            >
              Before
            </button>
            <button
              onClick={() => setViewMode('after')}
              className={`px-3 py-1 rounded text-xs ${viewMode === 'after' ? 'bg-blue-600' : 'bg-gray-700'}`}
            >
              After
            </button>
          </div>
        </div>
        
        {/* Layers */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold mb-2">Layers</h3>
          <label className="flex items-center gap-2 mb-2 text-sm">
            <input
              type="checkbox"
              checked={layers.damage}
              onChange={() => setLayers({...layers, damage: !layers.damage})}
            />
            Building Damage
          </label>
          <label className="flex items-center gap-2 mb-2 text-sm">
            <input
              type="checkbox"
              checked={layers.survivors}
              onChange={() => setLayers({...layers, survivors: !layers.survivors})}
            />
            Survivors
          </label>
          <label className="flex items-center gap-2 mb-2 text-sm">
            <input
              type="checkbox"
              checked={layers.hazards}
              onChange={() => setLayers({...layers, hazards: !layers.hazards})}
            />
            Hazards
          </label>
        </div>
        
        {/* Timeline */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
            <Clock className="w-4 h-4" />
            Timeline
          </h3>
          <input
            type="range"
            min="0"
            max="120"
            value={timeOffset}
            onChange={(e) => setTimeOffset(Number(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs mt-1">
            <span>0 min</span>
            <span>{timeOffset} min</span>
            <span>120 min</span>
          </div>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="w-full mt-2 px-3 py-2 bg-blue-600 rounded flex items-center justify-center gap-2"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
        </div>
        
        {/* Legend */}
        <div>
          <h3 className="text-sm font-semibold mb-2">Legend</h3>
          <div className="space-y-1 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4" style={{background: '#00ff00'}}></div>
              <span>No Damage</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4" style={{background: '#ffff00'}}></div>
              <span>Light Damage</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4" style={{background: '#ff8800'}}></div>
              <span>Moderate</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4" style={{background: '#ff0000'}}></div>
              <span>Severe</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4" style={{background: '#8b0000'}}></div>
              <span>Collapsed</span>
            </div>
            <hr className="my-2 border-gray-600" />
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span>Critical</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-orange-500"></div>
              <span>Medium</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span>Stable</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Center - Map */}
      <div className="flex-1 flex flex-col">
        {/* Metrics Bar */}
        <div className="bg-gray-800 p-4 flex gap-4">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            <div>
              <div className="text-xs text-gray-400">Coverage</div>
              <div className="text-lg font-bold">{metrics.coveragePercent}%</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5 text-green-400" />
            <div>
              <div className="text-xs text-gray-400">Survivors</div>
              <div className="text-lg font-bold">{metrics.survivorsFound}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <div>
              <div className="text-xs text-gray-400">Critical</div>
              <div className="text-lg font-bold">{metrics.criticalCount}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-yellow-400" />
            <div>
              <div className="text-xs text-gray-400">Avg Urgency</div>
              <div className="text-lg font-bold">{metrics.avgUrgency}%</div>
            </div>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            <span className="text-sm">Live Updates</span>
          </div>
        </div>
        
        {/* Map Canvas */}
        <div className="flex-1 relative bg-gray-900 overflow-auto">
          <canvas
            ref={canvasRef}
            width={600}
            height={600}
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasHover}
            className="cursor-crosshair"
          />
          
          {/* Hover tooltip */}
          {hoveredCell && (
            <div className="absolute top-4 left-4 bg-black bg-opacity-80 p-3 rounded text-xs max-w-xs pointer-events-none">
              <div className="font-bold mb-2">Cell {hoveredCell.id}</div>
              <div>Damage: {(hoveredCell.damageLevel * 100).toFixed(0)}%</div>
              <div>Buildings: {hoveredCell.postBuildings}/{hoveredCell.preBuildings}</div>
              <div>Population: {hoveredCell.population}</div>
              <div>Freshness: {(hoveredCell.freshnessScore * 100).toFixed(0)}%</div>
              {hoveredCell.hazards.length > 0 && (
                <div className="text-red-400">Hazards: {hoveredCell.hazards.join(', ')}</div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Right Sidebar - Survivors */}
      <div className="w-80 bg-gray-800 p-4 overflow-y-auto">
        <h2 className="text-lg font-bold mb-4">Survivors ({sortedSurvivors.length})</h2>
        
        <div className="space-y-2">
          {sortedSurvivors.slice(0, 20).map(survivor => (
            <div
              key={survivor.id}
              onClick={() => setSelectedSurvivor(survivor)}
              className={`p-3 rounded cursor-pointer transition ${
                selectedSurvivor?.id === survivor.id ? 'bg-blue-900' : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className={`px-2 py-1 rounded text-xs font-bold ${
                  survivor.status === 'CRITICAL' ? 'bg-red-600' :
                  survivor.status === 'MEDIUM' ? 'bg-orange-600' : 'bg-green-600'
                }`}>
                  {survivor.status}
                </span>
                <span className="text-xs text-gray-400">{survivor.id}</span>
              </div>
              
              <div className="mb-2">
                <div className="flex justify-between text-xs mb-1">
                  <span>Urgency</span>
                  <span>{(survivor.urgency * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-900 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      survivor.status === 'CRITICAL' ? 'bg-red-500' :
                      survivor.status === 'MEDIUM' ? 'bg-orange-500' : 'bg-green-500'
                    }`}
                    style={{width: `${survivor.urgency * 100}%`}}
                  ></div>
                </div>
              </div>
              
              <div className="text-xs space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-400">Location:</span>
                  <span>{survivor.lat.toFixed(2)}, {survivor.lon.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Detected:</span>
                  <span>{Math.floor((Date.now() - survivor.firstDetected) / 60000)}m ago</span>
                </div>
                <div className="flex gap-1 mt-2">
                  {survivor.detectionSources.map((source, idx) => (
                    <span key={idx} className="px-1 py-0.5 bg-gray-900 rounded text-xs">
                      {source}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DisasterResponseDemo;