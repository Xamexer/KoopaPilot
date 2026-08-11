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
-- SMW's native sprite clipping tables, indexed by the lower six bits of
-- SpriteTweakerB ($1662 + sprite slot). Reading the live tweaker byte also
-- captures sprites that change their clipping type while the level is running.
local SPRITE_CLIPPING_DISP_X = {
    0x02, 0x02, 0x10, 0x14, 0x00, 0x00, 0x01, 0x08,
    0xF8, 0xFE, 0x03, 0x06, 0x01, 0x00, 0x06, 0x02,
    0x00, 0xE8, 0xFC, 0xFC, 0x04, 0x00, 0xFC, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x00, 0x02, 0xE0, 0xF0,
    0xFC, 0xFC, 0x00, 0xF8, 0xF4, 0xF2, 0x00, 0xFC,
    0xF2, 0xF0, 0x02, 0x00, 0xF8, 0x04, 0x02, 0x02,
    0x08, 0x00, 0x00, 0x00, 0xFC, 0x03, 0x08, 0x00,
    0x08, 0x04, 0xF8, 0x00
}

local SPRITE_CLIPPING_WIDTH = {
    0x0C, 0x0C, 0x10, 0x08, 0x30, 0x50, 0x0E, 0x28,
    0x20, 0x14, 0x01, 0x03, 0x0D, 0x0F, 0x14, 0x24,
    0x0F, 0x40, 0x08, 0x08, 0x18, 0x0F, 0x18, 0x0C,
    0x0C, 0x0C, 0x0C, 0x0C, 0x0A, 0x1C, 0x30, 0x30,
    0x08, 0x08, 0x10, 0x20, 0x38, 0x3C, 0x20, 0x18,
    0x1C, 0x20, 0x0C, 0x10, 0x10, 0x08, 0x1C, 0x1C,
    0x10, 0x30, 0x30, 0x40, 0x08, 0x12, 0x34, 0x0F,
    0x20, 0x08, 0x20, 0x10
}

local SPRITE_CLIPPING_DISP_Y = {
    0x03, 0x03, 0xFE, 0x08, 0xFE, 0xFE, 0x02, 0x08,
    0xFE, 0x08, 0x07, 0x06, 0xFE, 0xFC, 0x06, 0xFE,
    0xFE, 0xE8, 0x10, 0x10, 0x02, 0xFE, 0xF4, 0x08,
    0x13, 0x23, 0x33, 0x43, 0x0A, 0xFD, 0xF8, 0xFC,
    0xE8, 0x10, 0x00, 0xE8, 0x20, 0x04, 0x58, 0xFC,
    0xE8, 0xFC, 0xF8, 0x02, 0xF8, 0x04, 0xFE, 0xFE,
    0xF2, 0xFE, 0xFE, 0xFE, 0xFC, 0x00, 0x08, 0xF8,
    0x10, 0x03, 0x10, 0x00
}

local SPRITE_CLIPPING_HEIGHT = {
    0x0A, 0x15, 0x12, 0x08, 0x0E, 0x0E, 0x18, 0x30,
    0x10, 0x1E, 0x02, 0x03, 0x16, 0x10, 0x14, 0x12,
    0x20, 0x40, 0x34, 0x74, 0x0C, 0x0E, 0x18, 0x45,
    0x3A, 0x2A, 0x1A, 0x0A, 0x30, 0x1B, 0x20, 0x12,
    0x18, 0x18, 0x10, 0x20, 0x38, 0x14, 0x08, 0x18,
    0x28, 0x1B, 0x13, 0x4C, 0x10, 0x04, 0x22, 0x20,
    0x1C, 0x12, 0x12, 0x12, 0x08, 0x20, 0x2E, 0x14,
    0x28, 0x0A, 0x10, 0x0D
}

local function signed_byte(value)
    if value >= 0x80 then
        return value - 0x100
    end
    return value
end

-- Reproduce SMW's 512-step circle lookup closely enough to locate parts that
-- rotate around a stationary sprite origin.
local function circle_offset(angle, radius)
    local radians = (angle & 0x1FF) * (2 * math.pi / 512)
    local sine = math.sin(radians)
    local magnitude = math.floor(math.abs(sine) * 256 + 0.000001)
    if magnitude > 256 then magnitude = 256 end

    local offset = math.floor(radius * magnitude / 256)
    if sine < 0 then return -offset end
    return offset
end

local function read_brown_chain_platform_footprint(slot, sprite_world_x, sprite_world_y)
    local angle = ((read_u8(0x1528 + slot) & 0x01) << 8) | read_u8(0x151C + slot)
    local platform_x = sprite_world_x - 0x50 + circle_offset(angle + 0x80, 0x50)
    local platform_y = sprite_world_y + circle_offset(angle, 0x50)

    -- Sprite 5F uses this custom rectangle for Mario/platform interaction.
    return {
        offset_x = platform_x - 0x18 - sprite_world_x,
        offset_y = platform_y - 0x0C - sprite_world_y,
        width = 0x40,
        height = 0x13
    }
end

local function read_turn_block_bridge_footprint(slot)
    local state = read_u8(0x00C2 + slot)
    local radius = read_u8(0x151C + slot)
    if (state & 0x02) ~= 0 then
        return { offset_x = 0, offset_y = -radius, width = 16, height = radius * 2 + 16 }
    end
    return { offset_x = -radius, offset_y = 0, width = radius * 2 + 16, height = 16 }
end

local function read_sprite_footprint(slot, sprite_id, sprite_world_x, sprite_world_y)
    if sprite_id == 0x5F then
        return read_brown_chain_platform_footprint(slot, sprite_world_x, sprite_world_y)
    end
    if sprite_id == 0x59 or sprite_id == 0x5A then
        return read_turn_block_bridge_footprint(slot)
    end

    local clipping_index = (read_u8(0x1662 + slot) & 0x3F) + 1
    return {
        offset_x = signed_byte(SPRITE_CLIPPING_DISP_X[clipping_index]),
        offset_y = signed_byte(SPRITE_CLIPPING_DISP_Y[clipping_index]),
        width = SPRITE_CLIPPING_WIDTH[clipping_index],
        height = SPRITE_CLIPPING_HEIGHT[clipping_index]
    }
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

        local sprite_world_x = (sx_high << 8) | sx_low
        local sprite_world_y = (sy_high << 8) | sy_low
        local hitbox = read_sprite_footprint(i, sprite_id, sprite_world_x, sprite_world_y)
        local hitbox_world_x = sprite_world_x + hitbox.offset_x
        local hitbox_world_y = sprite_world_y + hitbox.offset_y
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
    local ducking = read_u8(0x0073)
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
        ducking = ducking,
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

local GAME_MODE_OVERWORLD = 0x0E
local GAME_MODE_FADE_TO_LEVEL = 0x0F
local GAME_MODE_LOAD_LEVEL = 0x11
local GAME_MODE_FADE_IN_LEVEL = 0x13
local GAME_MODE_LEVEL = 0x14

--- Convert a Lunar Magic first-room ID into SMW's translevel identifier.
--- Only levels that can be entered as top-level overworld levels are valid.
local function level_id_to_translevel(level_id)
    if level_id >= 0x001 and level_id <= 0x024 then
        return level_id, 0
    end
    if level_id >= 0x101 and level_id <= 0x1DB then
        return level_id - 0x0DC, 1
    end
    return nil, nil
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
        if gm == GAME_MODE_OVERWORLD or gm == GAME_MODE_LEVEL then
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
    local translevel, target_submap = level_id_to_translevel(level_id)
    if not translevel then
        console.log(string.format(
            "[Emu #%d] ERROR: Level 0x%03X cannot be entered through SMW's full overworld loader",
            CONFIG.emulator_id, level_id
        ))
        return false
    end

    -- Ensure we are in a stable overworld or gameplay state.
    local gm = mainmemory.read_u8(0x0100)
    console.log(string.format("[Emu #%d]   Current game mode: 0x%02X", CONFIG.emulator_id, gm))

    if gm ~= GAME_MODE_OVERWORLD and gm ~= GAME_MODE_LEVEL then
        if not skip_intro() then
            console.log(string.format("[Emu #%d] ERROR: Cannot warp - game not ready", CONFIG.emulator_id))
            return false
        end
        gm = mainmemory.read_u8(0x0100)
    end

    console.log(string.format(
        "[Emu #%d]   Full-loading level 0x%03X via translevel 0x%02X",
        CONFIG.emulator_id, level_id, translevel
    ))

    -- Cancel death/end-level state before starting the same transition used by
    -- an overworld level entry. Game mode 0x11 then reloads headers, Map16,
    -- sprites, graphics, music, and Mario's entrance from ROM.
    mainmemory.write_u8(0x0071, 0x00)            -- player animation = normal
    mainmemory.write_u8(0x141A, 0x00)            -- first room, not a door/pipe sublevel
    mainmemory.write_u8(0x1493, 0x00)            -- clear end-level timer
    mainmemory.write_u8(0x13C6, 0x00)            -- clear cutscene ID
    mainmemory.write_u8(0x13CE, 0x00)            -- clear midway flag
    mainmemory.write_u8(0x13D2, 0x00)            -- clear boss/victory movement state

    -- $13BF identifies the translevel. $0109 is the essential forced-load
    -- override; without it the loader chooses a room from Mario's OW position.
    mainmemory.write_u8(0x13BF, translevel)
    mainmemory.write_u8(0x1F11, target_submap)
    mainmemory.write_u8(0x0109, translevel)

    -- Initialize the normal fade-out and enter the complete level-load chain:
    -- 0x0F -> 0x10 -> 0x11 -> 0x12 -> 0x13 -> 0x14.
    mainmemory.write_u8(0x0DAE, 0x0F)            -- full brightness
    mainmemory.write_u8(0x0DAF, 0x01)            -- fade/mosaic direction = out
    mainmemory.write_u8(0x0DB0, 0x00)            -- mosaic size
    mainmemory.write_u8(0x0DB1, 0x02)            -- same initial delay as OW entry
    mainmemory.write_u8(0x0100, GAME_MODE_FADE_TO_LEVEL)

    local wait_frames = 600
    local saw_full_load = false
    for i = 1, wait_frames do
        emu.frameadvance()
        local new_gm = mainmemory.read_u8(0x0100)

        if new_gm >= GAME_MODE_LOAD_LEVEL and new_gm <= GAME_MODE_FADE_IN_LEVEL then
            saw_full_load = true
        end

        if new_gm == GAME_MODE_LEVEL then
            local loaded_level_low = mainmemory.read_u8(0x1924)
            mainmemory.write_u8(0x0109, 0x00)
            mainmemory.write_u8(0x0071, 0x00)
            if not saw_full_load then
                console.log(string.format("[Emu #%d]   WARNING: Reached gameplay without observing full-load modes", CONFIG.emulator_id))
            end
            if loaded_level_low ~= (level_id & 0xFF) then
                console.log(string.format(
                    "[Emu #%d]   WARNING: Requested 0x%03X but loader reports low byte 0x%02X",
                    CONFIG.emulator_id, level_id, loaded_level_low
                ))
            end
            console.log(string.format("[Emu #%d]   Full level load complete (took %d frames)", CONFIG.emulator_id, i))
            return true
        end

        if new_gm == GAME_MODE_OVERWORLD or new_gm <= 0x07 then
            console.log(string.format("[Emu #%d]   Full level load failed in game mode 0x%02X", CONFIG.emulator_id, new_gm))
            break
        end
    end

    mainmemory.write_u8(0x0109, 0x00)
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
            -- Report the freshly loaded state before advancing any gameplay
            -- frames. RetroJet captures its reset state at this same boundary.
            goto continue
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
