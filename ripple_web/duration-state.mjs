const TIERS = ['major', 'small'];
const unavailable = (minute, hadConfirmed = false) => ({
  available: false, minute, ratio: null, streak: null, confirmed: false, hadConfirmed,
});

export function tierForAsset(duration, asset) {
  if (typeof asset !== 'string') return null;
  const matches = TIERS.filter((tier) => Array.isArray(duration?.groups?.[tier]?.assets)
    && duration.groups[tier].assets.includes(asset));
  return matches.length === 1 ? matches[0] : null;
}

// h=0 is the first point; ten qualifying points therefore confirm no earlier than h=9.
// A later rise ends current confirmation, while hadConfirmed records prior visible evidence.
export function durationStatus(duration, eventId, tier, minute) {
  if (!Number.isInteger(minute) || minute < 0 || minute > 120) {
    throw new RangeError('Duration minute must be an integer between 0 and 120.');
  }
  if (!duration || duration.threshold !== 1.2 || duration.confirmation_points !== 10
      || duration.reference_horizon_minutes !== 120 || !TIERS.includes(tier)) {
    return unavailable(minute);
  }
  const group = duration.events?.[eventId]?.groups?.[tier];
  if (group?.status !== 'valid' || !Array.isArray(group.curve)) return unavailable(minute);
  let streak = 0, hadConfirmed = false, ratio = null;
  for (let index = 0; index <= minute; index++) {
    ratio = group.curve[index];
    // Read only the known prefix. Invalid future points and stored future confirmations are irrelevant.
    if (!Number.isFinite(ratio) || ratio < 0) return unavailable(minute, hadConfirmed);
    streak = ratio <= duration.threshold ? streak + 1 : 0;
    if (streak >= duration.confirmation_points) hadConfirmed = true;
  }
  return {
    available: true, minute, ratio, streak,
    confirmed: streak >= duration.confirmation_points,
    hadConfirmed,
  };
}
