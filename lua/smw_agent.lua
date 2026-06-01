-- ============================================================================
-- KoopaPilot - BizHawk Lua Script
-- Communicates with Python training server via TCP sockets
-- ============================================================================

local json = {}

-- Minimal JSON encoder/decoder
function json.encode(val)
    local t = type(val)
    if t == "nil" then return "null"
    elseif t == "boolean" then return val and "true" or "false"
    elseif t == "number" then
        if val ~= val then return "null" end
        if val == math.huge or val == -math.huge then return "null" end
        return string.format("%.6g", val)
    elseif t == "string" then
        val = val:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('\n', '\\n'):gsub('\r', '\\r'):gsub('\t', '\\t')
        return '"' .. val .. '"'
    elseif t == "table" then
        local is_array = (#val > 0)
        if is_array then
            local parts = {}
            for i = 1, #val do
                parts[i] = json.encode(val[i])
            end
            return "[" .. table.concat(parts, ",") .. "]"
        else
            local parts = {}
            for k, v in pairs(val) do
                table.insert(parts, json.encode(tostring(k)) .. ":" .. json.encode(v))
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
    end
    return "null"
end

function json.decode(str)
    if str == nil or str == "" then return nil end
    -- Remove length prefix from BizHawk comm (format: "LENGTH payload")
    local payload = str:match("^%d+ (.+)$")
    if payload then str = payload end
    str = str:match("^%s*(.-)%s*$")
    if str == "" then return nil end

    local pos = 1
    local function skip_ws()
        pos = str:find("[^ \t\n\r]", pos) or (#str + 1)
    end

    local parse_value

    local function parse_string()
        pos = pos + 1 -- skip opening "
        local result = {}
        while pos <= #str do
            local c = str:sub(pos, pos)
            if c == '"' then pos = pos + 1; return table.concat(result) end
            if c == '\\' then
                pos = pos + 1
                c = str:sub(pos, pos)
                if c == 'n' then c = '\n'
                elseif c == 't' then c = '\t'
                elseif c == 'r' then c = '\r' end
            end
            table.insert(result, c)
            pos = pos + 1
        end
        return table.concat(result)
    end

    local function parse_number()
        local s = pos
        if str:sub(pos, pos) == '-' then pos = pos + 1 end
        while pos <= #str and str:sub(pos, pos):match("[%d%.eE%+%-]") do pos = pos + 1 end
        return tonumber(str:sub(s, pos - 1))
    end

    local function parse_object()
        pos = pos + 1 -- skip {
        local obj = {}
        skip_ws()
        if str:sub(pos, pos) == '}' then pos = pos + 1; return obj end
        while true do
            skip_ws()
            local key = parse_string()
            skip_ws()
            pos = pos + 1 -- skip :
            skip_ws()
            obj[key] = parse_value()
            skip_ws()
            if str:sub(pos, pos) == '}' then pos = pos + 1; return obj end
            pos = pos + 1 -- skip ,
        end
    end

    local function parse_array()
        pos = pos + 1 -- skip [
        local arr = {}
        skip_ws()
        if str:sub(pos, pos) == ']' then pos = pos + 1; return arr end
        while true do
            skip_ws()
            table.insert(arr, parse_value())
            skip_ws()
            if str:sub(pos, pos) == ']' then pos = pos + 1; return arr end
            pos = pos + 1 -- skip ,
        end
    end

    parse_value = function()
        skip_ws()
        local c = str:sub(pos, pos)
        if c == '"' then return parse_string()
        elseif c == '{' then return parse_object()
        elseif c == '[' then return parse_array()
        elseif c == 't' then pos = pos + 4; return true
        elseif c == 'f' then pos = pos + 5; return false
        elseif c == 'n' then pos = pos + 4; return nil
        else return parse_number()
        end
    end

    return parse_value()
end

-- ============================================================================
-- Configuration (received from server on connect)
-- ============================================================================
local CONFIG = {
    emulator_id = 0,
    frame_skip = 4,
    visibility = true,
    reward_display = true,
    button_input_display = true,
    mode = "training",
    speed_percent = 6400,
    level_load_mode = "savestate",
    levels = {},
    savestate_files = {},
    max_episode_steps = 4500,
    stagnation_timeout = 600,
    grid_size = 15,
    screenshot_dir = nil  -- Set during evaluation for video recording
}
local screenshot_frame_counter = 0

-- ============================================================================
-- Tile category mapping (Map16 tile number -> category)
-- ============================================================================
local TILE_EMPTY = 0
local TILE_WATER = 1
local TILE_COIN = 2
local TILE_VINE = 3
local TILE_MIDWAY = 4
local TILE_DANGEROUS = 5
local TILE_PIPE = 6
local TILE_QUESTION = 7
local TILE_TURNING = 8
local TILE_THROW = 9
local TILE_HALF_SOLID = 10
local TILE_SOLID = 11
local TILE_RAMP = 12
local TILE_DOOR = 13
local TILE_SLOPE_R = 14
local TILE_SLOPE_L = 15

local tile_lookup = {}

local function init_tile_lookup()
    -- Water: 0, 2
    for i = 0x000, 0x003 do tile_lookup[i] = TILE_WATER end
    -- Coins: 0x2A-0x2E
    for i = 0x02A, 0x02E do tile_lookup[i] = TILE_COIN end

    -- Vine: 0x006-0x00F, 0x010-0x01C
    for i = 0x006, 0x00F do tile_lookup[i] = TILE_VINE end
    for i = 0x010, 0x01C do tile_lookup[i] = TILE_VINE end

    -- Midway: 0x038
    tile_lookup[0x038] = TILE_MIDWAY

    -- Dangerous: 0x004, 0x005, 0x1FF, 0x1D2-0x1D7, 0x159-0x15C, 0x12F
    tile_lookup[0x004] = TILE_DANGEROUS
    tile_lookup[0x005] = TILE_DANGEROUS
    tile_lookup[0x1FF] = TILE_DANGEROUS
    for i = 0x1D2, 0x1D7 do tile_lookup[i] = TILE_DANGEROUS end
    for i = 0x159, 0x15C do tile_lookup[i] = TILE_DANGEROUS end
    tile_lookup[0x12F] = TILE_DANGEROUS

    -- Pipe: 0x133-0x13F
    for i = 0x133, 0x13F do tile_lookup[i] = TILE_PIPE end

    -- Question block: 0x021, 0x022, 0x029, 0x114, 0x117-0x11D, 0x11F-0x12B
    tile_lookup[0x021] = TILE_QUESTION
    tile_lookup[0x022] = TILE_QUESTION
    tile_lookup[0x029] = TILE_QUESTION
    tile_lookup[0x114] = TILE_QUESTION
    for i = 0x117, 0x11D do tile_lookup[i] = TILE_QUESTION end
    for i = 0x11F, 0x12B do tile_lookup[i] = TILE_QUESTION end

    -- Turning block: 0x11E
    tile_lookup[0x11E] = TILE_TURNING

    -- Throw block: 0x12E
    tile_lookup[0x12E] = TILE_THROW

    -- Half solid: 0x100-0x10C
    for i = 0x100, 0x10C do tile_lookup[i] = TILE_HALF_SOLID end

    -- Solid: 0x130, 0x132, 0x140-0x158, 0x14F-0x16D, 0x1C4-0x1C9
    tile_lookup[0x130] = TILE_SOLID
    tile_lookup[0x132] = TILE_SOLID
    for i = 0x140, 0x158 do tile_lookup[i] = TILE_SOLID end
    for i = 0x14F, 0x16D do tile_lookup[i] = TILE_SOLID end
    for i = 0x1C4, 0x1C9 do tile_lookup[i] = TILE_SOLID end

    -- Ramp: 0x1B4, 0x1B5
    tile_lookup[0x1B4] = TILE_RAMP
    tile_lookup[0x1B5] = TILE_RAMP

    -- Door: 0x01F, 0x020
    tile_lookup[0x01F] = TILE_DOOR
    tile_lookup[0x020] = TILE_DOOR

    -- Slopes right
    tile_lookup[0x1B6] = TILE_SLOPE_R
    for i = 0x16E, 0x181 do tile_lookup[i] = TILE_SLOPE_R end
    for i = 0x196, 0x19F do tile_lookup[i] = TILE_SLOPE_R end
    for i = 0x1AA, 0x1AE do tile_lookup[i] = TILE_SLOPE_R end
    for i = 0x1B8, 0x1B9 do tile_lookup[i] = TILE_SLOPE_R end
    for i = 0x1BC, 0x1BD do tile_lookup[i] = TILE_SLOPE_R end
    for i = 0x1C0, 0x1C1 do tile_lookup[i] = TILE_SLOPE_R end
    for i = 0x1CA, 0x1CB do tile_lookup[i] = TILE_SLOPE_R end

    -- Slopes left
    tile_lookup[0x1B7] = TILE_SLOPE_L
    for i = 0x182, 0x195 do tile_lookup[i] = TILE_SLOPE_L end
    for i = 0x1A0, 0x1A9 do tile_lookup[i] = TILE_SLOPE_L end
    for i = 0x1AF, 0x1B3 do tile_lookup[i] = TILE_SLOPE_L end
    for i = 0x1BA, 0x1BB do tile_lookup[i] = TILE_SLOPE_L end
    for i = 0x1BE, 0x1BF do tile_lookup[i] = TILE_SLOPE_L end
    for i = 0x1C2, 0x1C3 do tile_lookup[i] = TILE_SLOPE_L end
    for i = 0x1CC, 0x1CD do tile_lookup[i] = TILE_SLOPE_L end
end

init_tile_lookup()

-- ============================================================================
-- Tile category colors for overlay
-- ============================================================================
local TILE_COLORS = {
    [TILE_EMPTY]     = nil,
    [TILE_WATER]     = "#400000FF",
    [TILE_COIN]      = "#60FFD700",
    [TILE_VINE]      = "#4000AA00",
    [TILE_MIDWAY]    = "#60FF69B4",
    [TILE_DANGEROUS] = "#60FF0000",
    [TILE_PIPE]      = "#4000FF00",
    [TILE_QUESTION]  = "#60FFFF00",
    [TILE_TURNING]   = "#60FFA500",
    [TILE_THROW]     = "#60FF8C00",
    [TILE_HALF_SOLID]= "#408888FF",
    [TILE_SOLID]     = "#60AAAAAA",
    [TILE_RAMP]      = "#60888800",
    [TILE_DOOR]      = "#60FF00FF",
    [TILE_SLOPE_R]   = "#6000AAFF",
    [TILE_SLOPE_L]   = "#60AA00FF",
}

-- ============================================================================
-- Button input display colors
-- ============================================================================
local BUTTON_COLORS = {
    Right = "#00FF00",
    Left  = "#00FF00",
    Up    = "#00FF00",
    Down  = "#00FF00",
    A     = "#FF4444",
    B     = "#4444FF",
    Y     = "#FFFF00",
}

-- ============================================================================
-- RAM reading helpers
-- ============================================================================
local function read_u8(addr)
    return mainmemory.read_u8(addr)
end

local function read_u16(low_addr)
    return mainmemory.read_u16_le(low_addr)
end

local function read_s8(addr)
    return mainmemory.read_s8(addr)
end

-- ============================================================================
-- Get Map16 tile at a given level position
-- ============================================================================
local function get_map16_tile(level_x, level_y, is_vertical)
    local tile_x = math.floor(level_x / 16)
    local tile_y = math.floor(level_y / 16)

    local index
    if is_vertical then
        -- Vertical levels: use the complex page/screen system
        local screen = math.floor(tile_y / 16)
        local local_y = tile_y % 16
        local page = math.floor(screen / 2)
        index = (page << 8) | ((screen & 1) << 7) | (local_y << 4) | (tile_x & 0x0F)
    else
        -- Horizontal levels use SMW's screen-based tile layout.
        -- index = floor(tx / 16) * 0x1B0 + ty * 16 + (tx % 16)
        -- where 0x1B0 = 432 = bytes per screen (27 rows * 16 tiles?)
        -- Actually SMW uses 0x1B0 bytes per screen horizontally
        local screen = math.floor(tile_x / 16)
        local local_x = tile_x % 16
        index = screen * 0x1B0 + tile_y * 16 + local_x
    end

    -- Clamp index to valid range
    if index < 0 or index >= 0x4000 then return 0 end

    local low_byte = mainmemory.read_u8(0xC800 + index)
    -- High byte is in bank 7F ($7FC800)
    -- WRAM domain: bank 7E = 0x0000-0xFFFF, bank 7F = 0x10000-0x1FFFF
    -- So $7FC800 = WRAM offset 0x10000 + 0xC800 = 0x1C800
    local high_byte = 0
    pcall(function()
        high_byte = mainmemory.read_u8(0x1C800 + index)
    end)

    return (high_byte << 8) | low_byte
end

local function classify_tile(tile_num)
    return tile_lookup[tile_num] or TILE_EMPTY
end

-- ============================================================================
-- Build tile grid around Mario (size from CONFIG)
-- ============================================================================
local function build_tile_grid(mario_lx, mario_ly, camera_x, camera_y, is_vertical)
    local grid = {}
    local grid_positions = {} -- for overlay drawing
    local grid_size = CONFIG.grid_size or 15
    local half = math.floor(grid_size / 2)

    -- Build grid centered on Mario's position
    -- Note: mario_lx/ly are already centered (+8/+8 added in read_game_state)
    for dy = -half, half do
        local row = {}
        local pos_row = {}
        for dx = -half, half do
            -- Calculate tile pixel position relative to Mario
            -- Mario is centered, so tiles are at dx*16/dy*16 offset
            local px = mario_lx + dx * 16
            local py = mario_ly + dy * 16

            -- Screen position: subtract camera
            -- px/py are already centered (Mario's center position + offset)
            -- Subtract 8,8 to align tile with actual game tiles (they're read from top-left but displayed centered)
            local screen_x = px - camera_x - 8
            local screen_y = py - camera_y - 8
            
            -- Check if tile is on screen (0-255 x 0-223)
            -- Tiles completely offscreen are marked as empty
            local is_onscreen = (screen_x >= 0 and screen_x < 256 and 
                                 screen_y >= 0 and screen_y < 224)
            
            if is_onscreen then
                -- For grid data, read the tile at this position
                local tile_num = get_map16_tile(px, py, is_vertical)
                table.insert(row, classify_tile(tile_num))
            else
                -- Offscreen tiles are empty
                table.insert(row, TILE_EMPTY)
            end
            
            -- Store screen position for overlay drawing
            table.insert(pos_row, {sx = screen_x, sy = screen_y})
        end
        table.insert(grid, row)
        table.insert(grid_positions, pos_row)
    end
    return grid, grid_positions
end

-- ============================================================================
-- Read sprite data
-- ============================================================================
-- Keep a stable sprite footprint in the observation schema so existing
-- checkpoints remain compatible without embedding game-specific tables.
local function read_sprite_footprint()
    return { offset_x = 0, offset_y = 0, width = 16, height = 16 }
end

local function read_sprites(camera_x, camera_y)
    local sprites = {}
    for i = 0, 11 do
        local status = read_u8(0x14C8 + i)
        local sprite_id = read_u8(0x009E + i)
        local sx_low = read_u8(0x00E4 + i)
        local sx_high = read_u8(0x14E0 + i)
        local sy_low = read_u8(0x00D8 + i)
        local sy_high = read_u8(0x14D4 + i)

        local hitbox = read_sprite_footprint()
        local hitbox_world_x = ((sx_high << 8) | sx_low) + hitbox.offset_x
        local hitbox_world_y = ((sy_high << 8) | sy_low) + hitbox.offset_y
        local world_x = hitbox_world_x + hitbox.width / 2
        local world_y = hitbox_world_y + hitbox.height / 2
        local screen_x = world_x - camera_x
        local screen_y = world_y - camera_y

        -- Sprite speed (signed)
        local speed_x = read_s8(0x00B6 + i)
        local speed_y = read_s8(0x00AA + i)
        -- Misc/state byte (rotation angle for rotating platforms, general state for others)
        local misc_state = read_u8(0x00C2 + i)

        local active = (status >= 0x08) and 1 or 0

        table.insert(sprites, {
            active = active,
            id = sprite_id,
            status = status,
            screen_x = screen_x,
            screen_y = screen_y,
            world_x = world_x,
            world_y = world_y,
            speed_x = speed_x,
            speed_y = speed_y,
            misc_state = misc_state,
            hitbox_width = hitbox.width,
            hitbox_height = hitbox.height
        })
    end
    return sprites
end

-- ============================================================================
-- Read full game state
-- ============================================================================
local function read_game_state()
    local mario_x_screen = read_u16(0x007E)
    local mario_y_screen = read_u16(0x0080)
    local mario_x_level = read_u16(0x00D1)
    local mario_y_level = read_u16(0x00D3)
    local camera_x = read_u16(0x001A)
    local camera_y = read_u16(0x001C)
    local powerup = read_u8(0x0019)
    local in_water = read_u8(0x0075)
    local in_air = read_u8(0x0072)
    local on_ground = read_u8(0x13EF)
    local climbing = read_u8(0x0074)
    local player_anim = read_u8(0x0071)
    local coins = read_u8(0x0DBF)
    local lives = read_u8(0x0DBE)
    local game_mode = read_u8(0x0100)
    local sublevel = read_u8(0x141A)
    local num_screens = read_u8(0x005D)
    local screen_mode = read_u8(0x005B)
    local goal_reached = read_u8(0x1493)
    local mario_x_speed = read_s8(0x007B)  -- signed: -128 to 127
    local mario_y_speed = read_s8(0x007D)  -- signed: -128 to 127
    local is_vertical = (screen_mode & 0x01) ~= 0

    -- Center Mario's position (+8/+8 because Mario is 16x16)
    -- This ensures consistency with sprite positions and tile grid
    local mario_x_level_centered = mario_x_level + 8
    local mario_y_level_centered = mario_y_level + 8
    local mario_x_screen_centered = mario_x_screen + 8
    local mario_y_screen_centered = mario_y_screen + 8

    local tile_grid, tile_positions = build_tile_grid(
        mario_x_level_centered, mario_y_level_centered, camera_x, camera_y, is_vertical
    )
    local sprites = read_sprites(camera_x, camera_y)

    return {
        type = "state",
        mario_x_screen = mario_x_screen_centered,
        mario_y_screen = mario_y_screen_centered,
        mario_x_level = mario_x_level_centered,
        mario_y_level = mario_y_level_centered,
        camera_x = camera_x,
        camera_y = camera_y,
        powerup = powerup,
        in_water = in_water,
        in_air = in_air,
        on_ground = on_ground,
        climbing = climbing,
        player_anim = player_anim,
        coins = coins,
        lives = lives,
        game_mode = game_mode,
        sublevel = sublevel,
        num_screens = num_screens,
        goal_reached = goal_reached,
        mario_x_speed = mario_x_speed,
        mario_y_speed = mario_y_speed,
        is_vertical = is_vertical,
        tile_grid = tile_grid,
        sprites = sprites
    }, tile_positions
end

-- ============================================================================
-- Apply action from agent
-- ============================================================================
local function apply_action(action)
    if action == nil then return end
    -- In human mode, don't override player input
    if CONFIG.mode == "human" then return end

    -- action[1]=Right, [2]=Left, [3]=Up, [4]=Down
    -- action[5]=A (spinjump), [6]=B (jump), [7]=ReleaseY (Y always held unless released)
    joypad.set({
        Right = action[1] == 1,
        Left  = action[2] == 1,
        Up    = action[3] == 1,
        Down  = action[4] == 1,
        A     = action[5] == 1,
        B     = action[6] == 1,
        Y     = action[7] ~= 1,  -- Y always held, release only when ReleaseY=1
    }, 1)
end

-- ============================================================================
-- Overlay drawing
-- ============================================================================
local last_action = {0,0,0,0,0,0,0}
local total_reward = 0
local last_total_reward = 0
local last_reward_event = ""
local last_reward_event_timer = 0

local function draw_tile_overlay(tile_positions, tile_grid)
    if not CONFIG.visibility then return end
    if tile_positions == nil then return end

    for row = 1, #tile_grid do
        for col = 1, #tile_grid[row] do
            local cat = tile_grid[row][col]
            -- Only draw non-empty tiles.
            if cat ~= TILE_EMPTY then
                local color = TILE_COLORS[cat]
                if color and tile_positions[row] and tile_positions[row][col] then
                    local pos = tile_positions[row][col]
                    -- Draw if visible on screen
                    if pos.sx > -16 and pos.sx < 256 and pos.sy > -16 and pos.sy < 224 then
                        gui.drawRectangle(pos.sx, pos.sy, 15, 15, color, color)
                    end
                end
            end
        end
    end
end

local function draw_sprite_overlay(sprites)
    if not CONFIG.visibility then return end

    for i, sp in ipairs(sprites) do
        if sp.active == 1 then
            local sx = sp.screen_x
            local sy = sp.screen_y
            local width = sp.hitbox_width or 16
            local height = sp.hitbox_height or 16
            local left = math.floor(sx - width / 2)
            local top = math.floor(sy - height / 2)
            if left + width >= 0 and left < 256 and top + height >= 0 and top < 224 then
                gui.drawRectangle(left, top, width, height, "#FF00FFFF", "#3000FFFF")
                gui.pixelText(left, top - 7,
                    string.format("ID:%02X S:%X", sp.id, sp.status),
                    "#FFFFFF", "#000000")
            end
        end
    end
end

local function draw_reward_overlay()
    if not CONFIG.reward_display then return end

    gui.pixelText(170, 8, string.format("R: %.1f", total_reward), "#00FF00", "#000000")

    if last_reward_event_timer > 0 then
        gui.pixelText(170, 18, last_reward_event, "#FFFF00", "#000000")
        last_reward_event_timer = last_reward_event_timer - 1
    end
end

local function draw_input_overlay(action)
    if not CONFIG.button_input_display then return end
    if action == nil then return end

    local names = {"R", "L", "U", "D", "A", "B", "!Y"}
    local full_names = {"Right", "Left", "Up", "Down", "A", "B", "ReleaseY"}
    local base_x = 2
    local base_y = 210

    for i = 1, 7 do
        local pressed = action[i] == 1
        local color = pressed and BUTTON_COLORS[full_names[i]] or "#444444"
        local bg = pressed and "#000000" or nil
        gui.pixelText(base_x + (i - 1) * 14, base_y, names[i], color, bg)
    end
end

local function draw_emulator_id()
    gui.pixelText(240, 2, string.format("#%d", CONFIG.emulator_id), "#FF8800", "#000000")
end

-- ============================================================================
-- Level loading
-- ============================================================================

-- Translevel lookup: maps Lunar Magic level number to translevel number
-- Hardcoded fallback for common levels
local translevel_lookup = nil
local HARDCODED_TRANSLEVELS = {
    -- Yoshi's Island
    [0x103] = 0x07,  -- Yoshi's Island 3
    [0x104] = 0x08,  -- Yoshi's Island 4
    [0x105] = 0x09,  -- Yoshi's Island 1
    [0x106] = 0x0A,  -- Yoshi's Island 2
    [0x107] = 0x0B,  -- Iggy's Castle
    -- Donut Plains
    [0x10B] = 0x0C,  -- Donut Plains 1
    [0x10C] = 0x0D,  -- Donut Plains 2
    [0x10D] = 0x0E,  -- Donut Plains 3
    [0x10E] = 0x0F,  -- Donut Plains 4
    [0x10F] = 0x10,  -- Morton's Castle
    -- Vanilla Dome
    [0x115] = 0x16,  -- Vanilla Dome 1
    [0x116] = 0x17,  -- Vanilla Dome 2
    [0x11A] = 0x1B,  -- Vanilla Dome 4
    [0x11D] = 0x1E,  -- Lemmy's Castle
    -- Forest of Illusion
    [0x121] = 0x22,  -- Forest of Illusion 1
    [0x123] = 0x24,  -- Ludwig's Castle
}

--- Read SMW's translevel table from ROM and build a reverse lookup.
local function build_translevel_lookup()
    local rom_offset = 0x2EC00
    local domain = nil

    -- Try various domain names for ROM access
    for _, name in ipairs({"CARTROM", "ROM", "CARTRIDGE ROM", "Cart ROM"}) do
        local ok = pcall(function() memory.read_u8(0, name) end)
        if ok then
            domain = name
            break
        end
    end

    -- Start with hardcoded values
    local lookup = {}
    for lvl, tl in pairs(HARDCODED_TRANSLEVELS) do
        lookup[lvl] = tl
    end

    if not domain then
        console.log("WARNING: Cannot access ROM domain, using hardcoded translevels")
        return lookup
    end

    console.log("Reading translevel table from ROM (domain: " .. domain .. ")")
    
    -- Read ROM table and merge with hardcoded (ROM takes precedence if different)
    for i = 0, 0x7F do
        local ok, lo, hi = pcall(function()
            return memory.read_u8(rom_offset + i * 2, domain),
                   memory.read_u8(rom_offset + i * 2 + 1, domain)
        end)
        if ok then
            local level_num = lo | ((hi & 0x01) << 8)
            if level_num > 0 then
                lookup[level_num] = i
            end
        end
    end

    -- Log discovered levels for warp configuration checks.
    console.log("Available translevels:")
    for _, lv in ipairs({0x105, 0x106, 0x103, 0x104, 0x107, 0x10B, 0x115, 0x123}) do
        local tl = lookup[lv]
        if tl then
            console.log("  Level " .. string.format("0x%03X", lv)
                .. " -> translevel " .. string.format("0x%02X", tl))
        end
    end

    return lookup
end

--- Skip SMW's boot/intro/title/file-select to reach overworld or gameplay.
--- Presses Start+A to advance through menus. Returns true on success.
local function skip_intro()
    console.log(string.format("[Emu #%d] Skipping intro sequence...", CONFIG.emulator_id))
    local max_frames = 3000
    local frame = 0

    while frame < max_frames do
        local gm = mainmemory.read_u8(0x0100)

        -- Overworld or in-level = ready for warping
        if gm == 0x0E or gm == 0x14 then
            console.log(string.format("[Emu #%d] Game ready (mode 0x%02X)", CONFIG.emulator_id, gm))
            return true
        end

        -- Press Start + A to advance menus
        joypad.set({Start = true, A = true}, 1)
        emu.frameadvance()
        joypad.set({}, 1)
        for i = 1, 3 do emu.frameadvance() end
        frame = frame + 4
    end

    console.log("WARNING: skip_intro timed out (mode "
        .. string.format("0x%02X", mainmemory.read_u8(0x0100)) .. ")")
    return false
end

local function load_random_savestate()
    if #CONFIG.savestate_files > 0 then
        local idx = math.random(1, #CONFIG.savestate_files)
        savestate.load(CONFIG.savestate_files[idx])
        emu.frameadvance()
        return true
    end
    return false
end

local function warp_to_level(level_id)
    -- Ensure we are in a stable game state
    local gm = mainmemory.read_u8(0x0100)
    console.log(string.format("[Emu #%d]   Current game mode: 0x%02X", CONFIG.emulator_id, gm))

    if gm ~= 0x0E and gm ~= 0x14 then
        if not skip_intro() then
            console.log(string.format("[Emu #%d] ERROR: Cannot warp - game not ready", CONFIG.emulator_id))
            return
        end
        gm = mainmemory.read_u8(0x0100)
    end

    -- Resolve the translevel number from the ROM lookup table
    local translevel
    if translevel_lookup then
        translevel = translevel_lookup[level_id]
        if translevel then
            console.log(string.format("[Emu #%d]   Using translevel 0x%02X for level 0x%03X", CONFIG.emulator_id, translevel, level_id))
        else
            console.log(string.format("[Emu #%d]   WARNING: No translevel for level 0x%03X - using raw low byte", CONFIG.emulator_id, level_id))
            translevel = level_id & 0xFF
        end
    else
        translevel = level_id & 0xFF
        console.log(string.format("[Emu #%d]   No ROM lookup - using translevel 0x%02X", CONFIG.emulator_id, translevel))
    end

    -- If we're in a level (0x14), go back to overworld first
    if gm == 0x14 then
        console.log(string.format("[Emu #%d]   Exiting current level to overworld...", CONFIG.emulator_id))
        -- Press Start+Select to exit level
        for i = 1, 10 do
            joypad.set({Start = true, Select = true}, 1)
            emu.frameadvance()
        end
        joypad.set({}, 1)
        -- Wait for overworld
        for i = 1, 120 do
            emu.frameadvance()
            if mainmemory.read_u8(0x0100) == 0x0E then
                console.log(string.format("[Emu #%d]   Back to overworld", CONFIG.emulator_id))
                break
            end
        end
    end

    -- Set up level load via the "prepare level" subroutine
    -- This mimics what happens when you press A on a level dot
    mainmemory.write_u8(0x13BF, translevel)      -- destination translevel
    mainmemory.write_u8(0x0089, 0x00)            -- normal entrance
    mainmemory.write_u8(0x141A, 0x00)            -- reset sublevel count
    mainmemory.write_u8(0x0071, 0x00)            -- player anim = normal
    mainmemory.write_u8(0x13CE, 0x00)            -- no special flags
    mainmemory.write_u8(0x0D9B, 0x00)            -- main overworld
    
    -- Set game mode to "fade out to level" (0x0B)
    -- This is the state that triggers level loading
    mainmemory.write_u8(0x0100, 0x0B)
    
    -- SMW needs a few frames to process the mode change
    -- Wait through the fade sequence: 0x0B -> 0x0C -> 0x0D -> 0x14
    local wait_frames = 400
    for i = 1, wait_frames do
        emu.frameadvance()
        local new_gm = mainmemory.read_u8(0x0100)
        
        if new_gm == 0x14 then
            console.log(string.format("[Emu #%d]   Level loaded OK (took %d frames)", CONFIG.emulator_id, i))
            mainmemory.write_u8(0x0071, 0x00)
            return true
        end
        
        -- Detect failure early
        if new_gm == 0x0E then
            -- Back to overworld - loading was cancelled
            console.log(string.format("[Emu #%d]   Load cancelled, back to overworld", CONFIG.emulator_id))
            break
        end
        
        if new_gm <= 0x07 then
            console.log(string.format("[Emu #%d]   Warp failed (game mode 0x%02X)", CONFIG.emulator_id, new_gm))
            break
        end
    end
    
    -- Retry: try a different approach
    console.log(string.format("[Emu #%d]   Retrying with alternative method...", CONFIG.emulator_id))
    skip_intro()
    
    -- Alternative: Set up a "fake" level entry
    -- First, set Mario's position on overworld to a specific node
    mainmemory.write_u8(0x13BF, translevel)
    mainmemory.write_u8(0x0089, 0x00)
    mainmemory.write_u8(0x141A, 0x00)
    mainmemory.write_u8(0x0071, 0x00)
    mainmemory.write_u8(0x13CE, 0x00)
    mainmemory.write_u8(0x0100, 0x0B)
    
    for i = 1, wait_frames do
        emu.frameadvance()
        local new_gm = mainmemory.read_u8(0x0100)
        if new_gm == 0x14 then
            console.log(string.format("[Emu #%d]   Level loaded on retry (took %d frames)", CONFIG.emulator_id, i))
            mainmemory.write_u8(0x0071, 0x00)
            return true
        end
        if new_gm == 0x0E then
            break
        end
    end
    
    console.log(string.format("[Emu #%d]   WARNING: warp_to_level failed (mode 0x%02X)", CONFIG.emulator_id, mainmemory.read_u8(0x0100)))
    return false
end

local function load_random_level()
    if #CONFIG.levels > 0 then
        local idx = math.random(1, #CONFIG.levels)
        local level_id = CONFIG.levels[idx]
        console.log(string.format("[Emu #%d] Warping to level 0x%03X", CONFIG.emulator_id, level_id))
        warp_to_level(level_id)
        return true
    end
    return false
end

local current_level_index = 0

local function reset_episode()
    -- Try savestates first (most reliable)
    if CONFIG.level_load_mode == "savestate" then
        if load_random_savestate() then
            return
        end
        console.log("WARNING: No savestates available")
    end
    
    -- Try level warping if savestates failed or not configured
    if #CONFIG.levels > 0 then
        current_level_index = current_level_index + 1
        if current_level_index > #CONFIG.levels then
            current_level_index = 1
        end
        local level_id = CONFIG.levels[current_level_index]
        console.log(string.format("[Emu #%d] Warping to level 0x%03X", CONFIG.emulator_id, level_id))
        if warp_to_level(level_id) then
            return
        end
        console.log("WARNING: Level warp failed, trying fallback...")
    end
    
    -- Fallback: just skip intro and use whatever level we're on
    -- This ensures training can continue even if warping fails
    console.log("Falling back to skip_intro - will use current level")
    skip_intro()
    
    -- Wait a moment to ensure we're in a stable state
    for i = 1, 60 do
        emu.frameadvance()
    end
end

-- ============================================================================
-- Socket communication
-- ============================================================================
local connected = false
local retry_count = 0
local MAX_RETRIES = 50

local function try_connect()
    -- comm.socketServer connects to Python (Python is the server)
    if comm.socketServerIsConnected() then
        connected = true
        return true
    end
    return false
end

local function send_message(msg_table)
    if not connected then return nil end

    local msg_str = json.encode(msg_table)
    comm.socketServerSend(msg_str .. "\n")

    -- Retry response read in case BizHawk has a short receive timeout
    for attempt = 1, 5 do
        local response = comm.socketServerResponse()
        if response ~= nil and response ~= "" then
            return json.decode(response)
        end
    end
    return nil
end

local function wait_for_connection()
    console.log(string.format("[Emu] Connecting to Python server..."))
    while not connected and retry_count < MAX_RETRIES do
        if try_connect() then
            console.log(string.format("[Emu] Connected to server (will receive ID in handshake)"))
            break
        end
        retry_count = retry_count + 1
        for i = 1, 30 do emu.frameadvance() end
    end

    if not connected then
        console.log(string.format("[Emu] ERROR: Could not connect to server after %d retries", MAX_RETRIES))
        return false
    end

    -- Send handshake / receive config
    local handshake = {type = "handshake", emulator_id = CONFIG.emulator_id}
    local response = send_message(handshake)
    if response then
        if response.emulator_id ~= nil then CONFIG.emulator_id = response.emulator_id end
        if response.frame_skip then CONFIG.frame_skip = response.frame_skip end
        if response.visibility ~= nil then CONFIG.visibility = response.visibility end
        if response.reward_display ~= nil then CONFIG.reward_display = response.reward_display end
        if response.button_input_display ~= nil then CONFIG.button_input_display = response.button_input_display end
        if response.mode then CONFIG.mode = response.mode end
        if response.speed_percent then
            client.speedmode(response.speed_percent)
        end
        if response.level_load_mode then CONFIG.level_load_mode = response.level_load_mode end
        if response.levels then CONFIG.levels = response.levels end
        if response.savestate_files then CONFIG.savestate_files = response.savestate_files end
        if response.max_episode_steps then CONFIG.max_episode_steps = response.max_episode_steps end
        if response.stagnation_timeout then CONFIG.stagnation_timeout = response.stagnation_timeout end
        if response.grid_size then CONFIG.grid_size = response.grid_size end
        if response.sound_enabled ~= nil then client.SetSoundOn(response.sound_enabled) end
        if response.screenshot_dir then
            CONFIG.screenshot_dir = response.screenshot_dir
            screenshot_frame_counter = 0
            console.log(string.format("[Emu #%d] Screenshot recording to: %s", CONFIG.emulator_id, CONFIG.screenshot_dir))
        end

        console.log(string.format("[Emu #%d] Config received. Mode: %s", CONFIG.emulator_id, CONFIG.mode))
    end
    return true
end

-- ============================================================================
-- Parse emulator ID from command line args or socket port
-- ============================================================================
local function detect_emulator_id()
    -- The emulator ID is derived from the socket port
    -- Port = base_port + id, so id = port - base_port
    -- We'll receive it from the server in the handshake response
    CONFIG.emulator_id = 0
end

-- ============================================================================
-- Main loop
-- ============================================================================
local function main()
    detect_emulator_id()

    -- Build translevel lookup from ROM (before connecting, so no frames wasted)
    translevel_lookup = build_translevel_lookup()

    -- Disable sound and set speed for training
    client.SetSoundOn(false)
    emu.limitframerate(false)

    if not wait_for_connection() then
        console.log(string.format("[Emu] Exiting due to connection failure."))
        return
    end

    -- Initial episode setup
    reset_episode()

    local step_count = 0
    local current_action = nil

    while true do
        -- Check if still connected
        if not comm.socketServerIsConnected() then
            console.log(string.format("[Emu #%d] Lost connection to server. Exiting.", CONFIG.emulator_id))
            break
        end

        step_count = step_count + 1

        -- Read state and send to server
        local state, tile_positions = read_game_state()
        state.step = step_count

        local response = send_message(state)

        if response == nil then
            -- Connection lost
            console.log("No response from server. Retrying...")
            emu.frameadvance()
            goto continue
        end

        -- Handle response
        if response.type == "action" then
            current_action = response.action
            if response.total_reward then 
                -- Log significant reward changes.
                local reward_jump = response.total_reward - last_total_reward
                if math.abs(reward_jump) > 20 then
                    local event_info = response.reward_event or "no event"
                    local sign = reward_jump > 0 and "+" or ""
                    console.log(string.format("[Emu #%d] REWARD JUMP: %s%.1f (total: %.1f) - Event: %s", 
                        CONFIG.emulator_id, sign, reward_jump, response.total_reward, event_info))
                end
                -- Update tracking variables AFTER logging
                last_total_reward = response.total_reward
                total_reward = response.total_reward 
            end
            if response.reward_event and response.reward_event ~= "" then
                last_reward_event = response.reward_event
                last_reward_event_timer = 120 -- show for ~2 seconds
            end
        elseif response.type == "reset" then
            console.log(string.format("[Emu #%d] Episode reset (total reward was: %.1f)", CONFIG.emulator_id, total_reward))
            -- Update screenshot dir if provided (new episode)
            if response.screenshot_dir then
                CONFIG.screenshot_dir = response.screenshot_dir
                screenshot_frame_counter = 0
            elseif response.screenshot_dir == false then
                CONFIG.screenshot_dir = nil
            end
            reset_episode()
            step_count = 0
            total_reward = 0
            last_total_reward = 0
            current_action = nil
            last_reward_event = ""
            last_reward_event_timer = 0
            -- Don't skip frameadvance - fall through to normal frame handling
        elseif response.type == "close" then
            console.log(string.format("[Emu #%d] Server requested close.", CONFIG.emulator_id))
            break
        elseif response.type == "config_update" then
            if response.visibility ~= nil then CONFIG.visibility = response.visibility end
            if response.reward_display ~= nil then CONFIG.reward_display = response.reward_display end
            if response.button_input_display ~= nil then CONFIG.button_input_display = response.button_input_display end
            if response.speed_percent then client.speedmode(response.speed_percent) end
        end

        -- Apply action and advance frames (frame skip)
        if current_action then
            apply_action(current_action)
        end

        -- Draw overlays
        draw_emulator_id()
        if tile_positions then
            draw_tile_overlay(tile_positions, state.tile_grid)
        end
        draw_sprite_overlay(state.sprites or {})
        draw_reward_overlay()
        draw_input_overlay(current_action)

        -- Frame skip: repeat same action for N frames
        emu.frameadvance()

        -- Save screenshot during evaluation for video recording
        if CONFIG.screenshot_dir then
            local path = string.format("%s/frame_%06d.png", CONFIG.screenshot_dir, screenshot_frame_counter)
            client.screenshot(path)
            screenshot_frame_counter = screenshot_frame_counter + 1
        end

        for fs = 2, CONFIG.frame_skip do
            if current_action then
                apply_action(current_action)
            end
            -- Redraw all overlays each frame to prevent flicker
            draw_emulator_id()
            if tile_positions then
                draw_tile_overlay(tile_positions, state.tile_grid)
            end
            draw_sprite_overlay(state.sprites or {})
            draw_reward_overlay()
            draw_input_overlay(current_action)
            emu.frameadvance()
        end

        ::continue::
    end
end

-- Run
main()
