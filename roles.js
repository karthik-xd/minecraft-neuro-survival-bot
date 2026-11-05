'use strict';
// roles.js - Role-Based Access Control (RBAC) & Permission Engine for Minecraft Neuro-Survival Bot.
// Enforces hierarchical permissions across:
// 1. Supreme-Master: Top authority. Untouchable. Obey all orders including suicide ("kill yourself").
//    Only role permitted to assign roles via Minecraft in-game chat.
// 2. Master: Obey standard tasks (mine, gather, protect, mobs). CANNOT order suicide.
//    CANNOT order attacks on equal or higher ranked players (Supreme Master or Master).
// 3. Small Master: Limited authority. Collect resources/food and kill mobs. Limited player attacks.
// 4. Custom Roles & Permissions: Created, modified, and managed exclusively via Web UI.

const fs = require('fs');
const path = require('path');

const SYSTEM_ROLES = {
  SUPREME_MASTER: 'supreme-master',
  MASTER: 'master',
  SMALL_MASTER: 'small master',
  UNRANKED: 'unranked'
};

const DEFAULT_ROLE_DEFINITIONS = {
  [SYSTEM_ROLES.SUPREME_MASTER]: {
    id: SYSTEM_ROLES.SUPREME_MASTER,
    name: 'Supreme Master',
    priority: 100,
    canSuicide: true,
    canKillPlayers: 'all', // can order attacking any non-supreme player
    canKillMobs: true,
    canCollect: true,
    canBuildCraft: true,
    canAssignRoles: true, // can promote/assign roles in chat
    canNavigate: true,
    canStop: true,
    immuneToAttack: true, // Bot will NEVER attack Supreme Master
    description: 'Supreme Commander. Absolute obedience, suicide authority, role assignment.'
  },
  [SYSTEM_ROLES.MASTER]: {
    id: SYSTEM_ROLES.MASTER,
    name: 'Master',
    priority: 50,
    canSuicide: false, // Refuses self-destruction
    canKillPlayers: 'subordinate_only', // Cannot attack Supreme Master or other Masters
    canKillMobs: true,
    canCollect: true,
    canBuildCraft: true,
    canAssignRoles: false, // In chat, only supreme-master can assign
    canNavigate: true,
    canStop: true,
    immuneToAttack: true, // Bot will not attack Master unless ordered by Supreme Master
    description: 'High Commander. Controls survival, crafting, and defense. Cannot command suicide or attack superiors.'
  },
  [SYSTEM_ROLES.SMALL_MASTER]: {
    id: SYSTEM_ROLES.SMALL_MASTER,
    name: 'Small Master',
    priority: 25,
    canSuicide: false,
    canKillPlayers: 'unranked_only', // Only unranked players
    canKillMobs: true,
    canCollect: true, // Collect items/resources/food
    canBuildCraft: false,
    canAssignRoles: false,
    canNavigate: true, // Follow / protect
    canStop: true,
    immuneToAttack: false,
    description: 'Junior Commander. Resource gathering, food procurement, and mob clearing.'
  },
  [SYSTEM_ROLES.UNRANKED]: {
    id: SYSTEM_ROLES.UNRANKED,
    name: 'Unranked Player',
    priority: 0,
    canSuicide: false,
    canKillPlayers: false,
    canKillMobs: false,
    canCollect: false,
    canBuildCraft: false,
    canAssignRoles: false,
    canNavigate: false,
    canStop: false,
    immuneToAttack: false,
    description: 'Standard player with no bot commanding privileges.'
  }
};

class RoleManager {
  constructor(dataDir = null) {
    this.dataDir = dataDir || path.join(__dirname, 'data_bot');
    try { fs.mkdirSync(this.dataDir, { recursive: true }); } catch {}
    this.storageFile = path.join(this.dataDir, 'player_roles.json');

    this.playerRoles = {}; // { usernameLower: roleId }
    this.playerDisplayNames = {}; // { usernameLower: originalCase }
    this.customRoles = {}; // { roleId: roleDefinition }

    this.load();
  }

  load() {
    try {
      if (fs.existsSync(this.storageFile)) {
        const raw = fs.readFileSync(this.storageFile, 'utf8');
        const data = JSON.parse(raw);
        this.playerRoles = data.playerRoles || {};
        this.playerDisplayNames = data.playerDisplayNames || {};
        this.customRoles = data.customRoles || {};
      }
    } catch (e) {
      console.warn('[ROLE MANAGER] Error loading player_roles.json:', e.message);
    }
  }

  save() {
    try {
      const data = {
        updatedAt: new Date().toISOString(),
        playerRoles: this.playerRoles,
        playerDisplayNames: this.playerDisplayNames,
        customRoles: this.customRoles
      };
      fs.writeFileSync(this.storageFile, JSON.stringify(data, null, 2), 'utf8');
    } catch (e) {
      console.error('[ROLE MANAGER] Error saving player_roles.json:', e.message);
    }
  }

  // Canonical role ID normalizer
  normalizeRole(roleStr) {
    if (!roleStr) return SYSTEM_ROLES.UNRANKED;
    const clean = roleStr.toLowerCase().trim().replace(/[-_]/g, ' ');
    if (clean === 'supreme master' || clean === 'suprememaster' || clean === 'supreme') return SYSTEM_ROLES.SUPREME_MASTER;
    if (clean === 'master') return SYSTEM_ROLES.MASTER;
    if (clean === 'small master' || clean === 'smallmaster' || clean === 'junior master' || clean === 'small') return SYSTEM_ROLES.SMALL_MASTER;

    // Check custom roles
    for (const rId of Object.keys(this.customRoles)) {
      if (rId.toLowerCase().replace(/[-_]/g, ' ') === clean) return rId;
    }
    return roleStr.toLowerCase().trim();
  }

  // Get active role id for a player
  getPlayerRole(username) {
    if (!username) return SYSTEM_ROLES.UNRANKED;
    const u = username.toLowerCase().trim();
    return this.playerRoles[u] || SYSTEM_ROLES.UNRANKED;
  }

  // Get full definition and permissions for a role ID
  getRoleDefinition(roleId) {
    const norm = this.normalizeRole(roleId);
    if (this.customRoles[norm]) return this.customRoles[norm];
    if (DEFAULT_ROLE_DEFINITIONS[norm]) return DEFAULT_ROLE_DEFINITIONS[norm];
    return DEFAULT_ROLE_DEFINITIONS[SYSTEM_ROLES.UNRANKED];
  }

  // Get player's active permissions object
  getPlayerPermissions(username) {
    const roleId = this.getPlayerRole(username);
    return this.getRoleDefinition(roleId);
  }

  // Assign role to player
  setPlayerRole(username, roleName, assignedBy = null, isFromUI = false) {
    if (!username || typeof username !== 'string') {
      return { success: false, error: 'Invalid player username' };
    }
    const u = username.toLowerCase().trim();
    const orig = username.trim();
    const targetRole = this.normalizeRole(roleName);

    // Verify role exists
    const roleDef = this.getRoleDefinition(targetRole);
    if (!roleDef || (targetRole === SYSTEM_ROLES.UNRANKED && roleName.toLowerCase() !== 'unranked')) {
      return { success: false, error: `Role "${roleName}" does not exist` };
    }

    // Permission check if assigned via in-game chat
    if (!isFromUI && assignedBy) {
      const assignerPerms = this.getPlayerPermissions(assignedBy);
      if (!assignerPerms.canAssignRoles) {
        return {
          success: false,
          error: `Permission denied: Only Supreme Master can assign roles in Minecraft chat!`
        };
      }
    }

    this.playerRoles[u] = targetRole;
    this.playerDisplayNames[u] = orig;
    this.save();

    return {
      success: true,
      player: orig,
      role: targetRole,
      roleName: roleDef.name,
      priority: roleDef.priority
    };
  }

  // Remove role assignment
  removePlayerRole(username) {
    if (!username) return { success: false };
    const u = username.toLowerCase().trim();
    delete this.playerRoles[u];
    delete this.playerDisplayNames[u];
    this.save();
    return { success: true, player: username };
  }

  // Create or update a custom role (UI only)
  createCustomRole(roleKey, config = {}) {
    if (!roleKey || typeof roleKey !== 'string') {
      return { success: false, error: 'Invalid role key' };
    }
    const cleanKey = roleKey.toLowerCase().trim().replace(/[^a-z0-9_-]/g, '_');
    if (Object.values(SYSTEM_ROLES).includes(cleanKey)) {
      return { success: false, error: `Cannot overwrite built-in system role "${cleanKey}"` };
    }

    const priority = Math.max(1, Math.min(99, parseInt(config.priority || '30', 10)));
    const roleDef = {
      id: cleanKey,
      name: config.name || cleanKey.toUpperCase(),
      priority,
      canSuicide: !!config.canSuicide,
      canKillPlayers: config.canKillPlayers === true || config.canKillPlayers === 'all' ? 'all' : (config.canKillPlayers || false),
      canKillMobs: config.canKillMobs !== undefined ? !!config.canKillMobs : true,
      canCollect: config.canCollect !== undefined ? !!config.canCollect : true,
      canBuildCraft: !!config.canBuildCraft,
      canAssignRoles: !!config.canAssignRoles,
      canNavigate: config.canNavigate !== undefined ? !!config.canNavigate : true,
      canStop: config.canStop !== undefined ? !!config.canStop : true,
      immuneToAttack: !!config.immuneToAttack,
      description: config.description || `Custom role created via Web UI.`
    };

    this.customRoles[cleanKey] = roleDef;
    this.save();

    return { success: true, role: roleDef };
  }

  // Delete custom role (UI only)
  deleteCustomRole(roleKey) {
    const cleanKey = (roleKey || '').toLowerCase().trim();
    if (this.customRoles[cleanKey]) {
      delete this.customRoles[cleanKey];
      // Revert any players with this role to unranked
      for (const [u, r] of Object.entries(this.playerRoles)) {
        if (r === cleanKey) delete this.playerRoles[u];
      }
      this.save();
      return { success: true, deleted: cleanKey };
    }
    return { success: false, error: `Custom role "${roleKey}" not found` };
  }

  // Check if sender has authority to order a task
  canExecuteTask(sender, intent, targetPlayer = null) {
    if (!sender) {
      return { allowed: true, role: 'system' }; // Internal server / UI orders bypass sender check
    }

    const perms = this.getPlayerPermissions(sender);
    const roleId = this.getPlayerRole(sender);

    // 1. Suicide task
    if (intent === 'suicide') {
      if (!perms.canSuicide) {
        return {
          allowed: false,
          reason: `Refused! Only Supreme Master can order my termination.`
        };
      }
      return { allowed: true, role: roleId };
    }

    // 2. Kill Player combat task
    if (intent === 'kill' && targetPlayer) {
      const targetRole = this.getPlayerRole(targetPlayer);
      const targetPerms = this.getRoleDefinition(targetRole);

      // Bot NEVER attacks a Supreme Master under any circumstance
      if (targetRole === SYSTEM_ROLES.SUPREME_MASTER || targetPerms.id === SYSTEM_ROLES.SUPREME_MASTER) {
        return {
          allowed: false,
          reason: `Cannot attack Supreme Master ${targetPlayer}! Supreme Masters are untouchable.`
        };
      }

      // If sender has 'all' player killing permissions (Supreme Master)
      if (perms.canKillPlayers === 'all' || perms.priority >= 100) {
        return { allowed: true, role: roleId };
      }

      // Master hierarchy rule: Master cannot kill equal or higher roles (Supreme Master or another Master)
      if (perms.canKillPlayers === 'subordinate_only') {
        if (targetPerms.priority >= perms.priority) {
          return {
            allowed: false,
            reason: `Permission denied! Master cannot order attack on higher or equal role player ${targetPlayer} (${targetPerms.name}).`
          };
        }
        return { allowed: true, role: roleId };
      }

      // Small Master rule: Can only attack unranked players
      if (perms.canKillPlayers === 'unranked_only' || roleId === SYSTEM_ROLES.SMALL_MASTER) {
        if (targetRole !== SYSTEM_ROLES.UNRANKED && targetPerms.priority > 0) {
          return {
            allowed: false,
            reason: `Permission denied! Small Master can only order attacks on unranked players (target ${targetPlayer} is ${targetPerms.name}).`
          };
        }
        return { allowed: true, role: roleId };
      }

      if (!perms.canKillPlayers) {
        return {
          allowed: false,
          reason: `Permission denied! You do not have permission to order player assassinations.`
        };
      }
    }

    // 3. Kill Mob combat task
    if (intent === 'kill') {
      if (!perms.canKillMobs) {
        return {
          allowed: false,
          reason: `Permission denied! Role ${perms.name} cannot order mob hunts.`
        };
      }
      return { allowed: true, role: roleId };
    }

    // 4. Mining, Gathering, Food, Resource Collection
    if (intent === 'mine' || intent === 'food' || intent === 'collect' || intent === 'chest_query') {
      if (!perms.canCollect) {
        return {
          allowed: false,
          reason: `Permission denied! Role ${perms.name} cannot order resource gathering.`
        };
      }
      return { allowed: true, role: roleId };
    }

    // 5. Crafting & Building
    if (intent === 'craft' || intent === 'build' || intent === 'farm') {
      if (!perms.canBuildCraft && perms.priority < 50) {
        return {
          allowed: false,
          reason: `Permission denied! Role ${perms.name} cannot command construction or crafting.`
        };
      }
      return { allowed: true, role: roleId };
    }

    // 6. Navigation (Follow, Protect, Goto)
    if (intent === 'protect' || intent === 'goto' || intent === 'find_village') {
      if (!perms.canNavigate) {
        return {
          allowed: false,
          reason: `Permission denied! Role ${perms.name} cannot issue navigation orders.`
        };
      }
      return { allowed: true, role: roleId };
    }

    // 7. Stop / Cancel
    if (intent === 'stop') {
      if (!perms.canStop && perms.priority === 0) {
        return {
          allowed: false,
          reason: `Permission denied! You cannot halt bot missions.`
        };
      }
      return { allowed: true, role: roleId };
    }

    // 8. In-game role assignment
    if (intent === 'assign_role') {
      if (!perms.canAssignRoles) {
        return {
          allowed: false,
          reason: `Permission denied! Only Supreme Master can assign player roles in Minecraft chat.`
        };
      }
      return { allowed: true, role: roleId };
    }

    return { allowed: true, role: roleId };
  }

  // Get full state snapshot for API / UI
  getStateSnapshot() {
    return {
      systemRoles: DEFAULT_ROLE_DEFINITIONS,
      customRoles: this.customRoles,
      playerRoles: Object.entries(this.playerRoles).map(([u, r]) => {
        const def = this.getRoleDefinition(r);
        return {
          username: this.playerDisplayNames[u] || u,
          roleId: r,
          roleName: def.name,
          priority: def.priority,
          canSuicide: !!def.canSuicide,
          canKillPlayers: def.canKillPlayers,
          description: def.description
        };
      })
    };
  }
}

module.exports = {
  RoleManager,
  SYSTEM_ROLES,
  DEFAULT_ROLE_DEFINITIONS
};
