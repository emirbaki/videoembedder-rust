//! The indexing plan. How one video is cut into records.

use anyhow::{Result, bail};

/// The two ways the app cuts a video.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// One record holds a window: several frames and the window audio.
    Window,
    /// One record holds a 0.25 s frame, or one second of audio.
    Fine,
}

impl Mode {
    /// The stored name of the mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Window => "window",
            Mode::Fine => "fine",
        }
    }

    /// Read the mode from its stored name.
    pub fn parse(word: &str) -> Option<Self> {
        match word {
            "window" => Some(Mode::Window),
            "fine" => Some(Mode::Fine),
            _ => None,
        }
    }
}

/// A named plan. The three Google presets, plus the fine grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    Fine,
    Object,
    Action,
    Speech,
}

impl Preset {
    /// Every preset, in the order the page shows them.
    pub const ALL: [Preset; 4] = [Preset::Fine, Preset::Object, Preset::Action, Preset::Speech];

    /// The stored name of the preset.
    pub fn as_str(self) -> &'static str {
        match self {
            Preset::Fine => "fine",
            Preset::Object => "object",
            Preset::Action => "action",
            Preset::Speech => "speech",
        }
    }

    /// Read a preset from a word. The word is lowercased first.
    pub fn parse(word: &str) -> Option<Self> {
        match word.to_ascii_lowercase().as_str() {
            "fine" => Some(Preset::Fine),
            "object" => Some(Preset::Object),
            "action" => Some(Preset::Action),
            "speech" => Some(Preset::Speech),
            _ => None,
        }
    }
}

/// How one video is cut.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Plan {
    pub mode: Mode,
    /// The seconds of one window.
    pub window: f64,
    /// The seconds between two window starts.
    pub stride: f64,
    /// The frame count in one window.
    pub frames: usize,
    /// The window record holds the window audio.
    pub audio: bool,
}

/// The shortest window the plan accepts, in seconds.
pub const MIN_WINDOW: f64 = 1.0;
/// The longest window the plan accepts, in seconds.
pub const MAX_WINDOW: f64 = 10.0;

impl Plan {
    /// The plan of one preset.
    pub fn of(preset: Preset) -> Plan {
        match preset {
            Preset::Fine => Plan {
                mode: Mode::Fine,
                window: 0.25,
                stride: 0.25,
                frames: 1,
                audio: true,
            },
            // The probe measured this shape at 10 hits of 10 with the window
            // audio, against 6 of 10 without it. Google's Object preset holds
            // no audio, so this value departs from that table.
            Preset::Object => Plan {
                mode: Mode::Window,
                window: 2.0,
                stride: 1.0,
                frames: 2,
                audio: true,
            },
            Preset::Action => Plan {
                mode: Mode::Window,
                window: 4.0,
                stride: 2.0,
                frames: 4,
                audio: false,
            },
            Preset::Speech => Plan {
                mode: Mode::Window,
                window: 6.0,
                stride: 3.0,
                frames: 6,
                audio: true,
            },
        }
    }

    /// The plan a video takes when it holds none.
    pub fn default_plan() -> Plan {
        Plan::of(Preset::Object)
    }

    /// The frames a second inside one window.
    pub fn frame_rate(&self) -> f64 {
        self.frames as f64 / self.window
    }

    /// The start of every window in one video.
    ///
    /// A window that would reach past the end does not start. The count follows
    /// from the duration, the window, and the stride.
    pub fn window_starts(&self, duration: f64) -> Vec<f64> {
        let mut starts = Vec::new();
        let mut t = 0.0;
        while t + self.window <= duration {
            starts.push(t);
            t += self.stride;
        }
        starts
    }

    /// Set one or more plan values over this plan.
    ///
    /// The words are `window=4 frames=4 audio=on`. A window that the user names
    /// sets the stride to half of that window, so the windows overlap.
    pub fn set_words(&self, words: &str) -> Result<Plan> {
        let mut plan = *self;
        let mut named_window = false;
        let mut named_stride = false;
        for word in words.split_whitespace() {
            let Some((key, value)) = word.split_once('=') else {
                bail!("{word} is not a plan value");
            };
            match key {
                "window" => {
                    plan.window = parse_seconds(value)?;
                    named_window = true;
                }
                "stride" => {
                    plan.stride = parse_seconds(value)?;
                    named_stride = true;
                }
                "frames" => {
                    plan.frames = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("{value} is not a frame count"))?;
                }
                "audio" => plan.audio = parse_flag(value)?,
                _ => bail!("{key} is not a plan value"),
            }
        }
        if plan.window < MIN_WINDOW || plan.window > MAX_WINDOW {
            bail!("the window must be {MIN_WINDOW} to {MAX_WINDOW} seconds");
        }
        if plan.frames < 1 || plan.frames as f64 > plan.window {
            bail!("the frame count must be 1 to {}", plan.window as usize);
        }
        if named_window && !named_stride {
            plan.stride = plan.window / 2.0;
        }
        if plan.stride <= 0.0 {
            bail!("the stride must be more than zero");
        }
        plan.mode = Mode::Window;
        Ok(plan)
    }
}

/// Read a seconds value.
fn parse_seconds(value: &str) -> Result<f64> {
    let seconds: f64 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("{value} is not a number"))?;
    if !seconds.is_finite() {
        bail!("{value} is not a number");
    }
    Ok(seconds)
}

/// Read an on or off word.
fn parse_flag(value: &str) -> Result<bool> {
    match value {
        "on" | "yes" | "1" => Ok(true),
        "off" | "no" | "0" => Ok(false),
        _ => bail!("{value} is not on or off"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preset_holds_its_values() {
        let object = Plan::of(Preset::Object);
        assert_eq!(object.mode, Mode::Window);
        assert_eq!(object.window, 2.0);
        assert_eq!(object.stride, 1.0);
        assert_eq!(object.frames, 2);
        assert!(object.audio);

        let speech = Plan::of(Preset::Speech);
        assert_eq!(speech.window, 6.0);
        assert_eq!(speech.frames, 6);
        assert!(speech.audio);

        let fine = Plan::of(Preset::Fine);
        assert_eq!(fine.mode, Mode::Fine);
    }

    /// The default plan is the Object preset.
    #[test]
    fn the_default_plan_is_object() {
        assert_eq!(Plan::default_plan(), Plan::of(Preset::Object));
    }

    #[test]
    fn the_frame_rate_follows_from_the_window() {
        assert_eq!(Plan::of(Preset::Object).frame_rate(), 1.0);
        assert_eq!(Plan::of(Preset::Action).frame_rate(), 1.0);
        assert_eq!(Plan::of(Preset::Speech).frame_rate(), 1.0);
    }

    /// A name reads back to its preset, and a foreign word reads as nothing.
    #[test]
    fn a_preset_name_round_trips() {
        for preset in Preset::ALL {
            assert_eq!(Preset::parse(preset.as_str()), Some(preset));
        }
        assert_eq!(Preset::parse("object"), Some(Preset::Object));
        assert_eq!(Preset::parse("OBJECT"), Some(Preset::Object));
        assert_eq!(Preset::parse("fast"), None);
    }

    /// A 30 s video gives 29 windows at 2 s on a 1 s stride.
    #[test]
    fn the_window_starts_cover_a_thirty_second_video() {
        let plan = Plan::of(Preset::Object);
        let starts = plan.window_starts(30.0);
        assert_eq!(starts.len(), 29);
        assert_eq!(starts[0], 0.0);
        assert_eq!(starts[1], 1.0);
        assert_eq!(*starts.last().unwrap(), 28.0);
    }

    /// A video shorter than one window gives no window.
    #[test]
    fn a_short_video_gives_no_window() {
        let plan = Plan::of(Preset::Object);
        assert!(plan.window_starts(1.5).is_empty());
    }

    #[test]
    fn a_value_word_sets_one_plan_value() {
        let base = Plan::of(Preset::Object);
        let asked = base.set_words("window=4 frames=4 audio=on").unwrap();
        assert_eq!(asked.window, 4.0);
        assert_eq!(asked.frames, 4);
        assert!(asked.audio);
        // The stride follows the window when the user does not name it.
        assert_eq!(asked.stride, 2.0);
        assert_eq!(asked.mode, Mode::Window);
    }

    /// A bad value names the fault.
    #[test]
    fn a_bad_value_is_refused() {
        let base = Plan::of(Preset::Object);
        assert!(base.set_words("window=0").is_err(), "a zero window");
        assert!(base.set_words("window=99").is_err(), "a window past 10 s");
        assert!(
            base.set_words("frames=5 window=2").is_err(),
            "frames past the window"
        );
        assert!(base.set_words("candy=1").is_err(), "an unknown key");
    }
}
