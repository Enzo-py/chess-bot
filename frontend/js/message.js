const ctx = {

}

function read_message(event) {
    content = JSON.parse(event.data);
    switch (content.type) {
        case "popup":
            read_popup_message(content.data);
            break;
        case "toast":
            read_toast_message(content.data);
            break;
        case "game-started":
            read_game_started(content.data.message);
            break;
        case "confirm-move":
            game_state.turn = game_state.turn === "w" ? "b" : "w";
            read_confirm_move(content.data.message);
            break;
        default:
            return content
    }

    return content
}

function read_popup_message(data) {
    // data
}

function read_toast_message(data) {
    // data
}

function read_game_started(data) {
    // data
    draw_game(data.FEN);
    update_game_state(data.FEN);
    new Audio('../media/game-start.mp3').play();
}

function read_confirm_move(data) {
    // data
    update_game_state(data.FEN);
}
