use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::base::{AlgoType, RetrieveMode, Transition};
use crate::component::{Component, ComponentEvent};
use crate::states::compare_mode::SelectAlgoState;
use crate::states::main_menu::MainState;
use crate::states::retrieve_mode::RetrieveModeSelectState;

#[test]
fn test_main_menu_r_goes_to_retrieve_mode() {
    let mut state = MainState;
    let t = state.handle_event(ComponentEvent::Key(KeyEvent::new(
        KeyCode::Char('r'),
        KeyModifiers::NONE,
    )));
    assert!(matches!(t, Transition::ToRetrieveModeSelect));
}

#[test]
fn test_main_menu_d_goes_to_select_algo() {
    let mut state = MainState;
    let t = state.handle_event(ComponentEvent::Key(KeyEvent::new(
        KeyCode::Char('d'),
        KeyModifiers::NONE,
    )));
    assert!(matches!(t, Transition::ToSelectAlgo));
}

#[test]
fn test_main_menu_colon_opens_command() {
    let mut state = MainState;
    let t = state.handle_event(ComponentEvent::Key(KeyEvent::new(
        KeyCode::Char(':'),
        KeyModifiers::NONE,
    )));
    assert!(matches!(t, Transition::ToCommand(_)));
}

#[test]
fn test_retrieve_mode_select_embedding() {
    let mut state = RetrieveModeSelectState::new();
    let t = state.handle_event(ComponentEvent::Key(KeyEvent::new(
        KeyCode::Char('e'),
        KeyModifiers::NONE,
    )));
    assert!(
        matches!(
            t,
            Transition::ToSelectDataset(AlgoType::Retrieve(RetrieveMode::Embedding))
        ),
        "pressing e should select embedding mode"
    );
}

#[test]
fn test_select_algo_enter_r() {
    let mut state = SelectAlgoState::new();
    let t = state.handle_event(ComponentEvent::Key(KeyEvent::new(
        KeyCode::Char('r'),
        KeyModifiers::NONE,
    )));
    assert!(matches!(t, Transition::ToSelectCompareDataset));
}
